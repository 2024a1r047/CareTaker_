/**
 * med_worker.js — medicine recognition in a background thread.
 *
 * Fall detection (MoveNet) runs on the graphics chip in the page. This worker runs the
 * medicine recogniser (MobileNet fingerprints) on the CPU with TensorFlow.js' WebAssembly
 * backend, so both work at the same time and the fall camera never has to wait.
 *
 * Messages in:
 *   {type:'init', modelUrl, wasmBase}                 -> {type:'ready', backend} | {type:'error', message}
 *   {type:'embed', id, w, h, data(ArrayBuffer RGBA), boxes:[[y1,x1,y2,x2],...]}
 *                                                     -> {type:'embeds', id, dim, data(Float32Array buffer), ms}
 * The fingerprints are the same as the page's embed(): MobileNet v2 0.5 embedding, L2-normalised.
 */
/* global tf, mobilenet, importScripts */
'use strict';

const CDN = 'https://cdn.jsdelivr.net/npm/';
let net = null;

function loadLibs() {
    importScripts(
        CDN + '@tensorflow/tfjs-core@4.17.0/dist/tf-core.min.js',
        CDN + '@tensorflow/tfjs-converter@4.17.0/dist/tf-converter.min.js',
        CDN + '@tensorflow/tfjs-backend-wasm@4.17.0/dist/tf-backend-wasm.min.js',
        CDN + '@tensorflow-models/mobilenet@2.1.1/dist/mobilenet.min.js'
    );
}

async function init(msg) {
    if (typeof tf === 'undefined') loadLibs();
    tf.wasm.setWasmPaths(msg.wasmBase || (CDN + '@tensorflow/tfjs-backend-wasm@4.17.0/dist/'));
    await tf.setBackend('wasm');
    await tf.ready();
    net = await mobilenet.load({ version: 2, alpha: 0.5, modelUrl: msg.modelUrl });
    // warm-up so the first real scan is fast
    tf.tidy(() => net.infer(tf.zeros([1, 224, 224, 3]), true));
    return tf.getBackend();
}

function embed(msg) {
    const { w, h, data, boxes } = msg;
    const t0 = Date.now();
    const out = tf.tidy(() => {
        const rgba = tf.tensor3d(new Uint8Array(data), [h, w, 4], 'int32');
        const rgb = tf.slice(rgba, [0, 0, 0], [h, w, 3]);
        const img = tf.expandDims(tf.cast(rgb, 'float32'), 0);
        const crops = tf.image.cropAndResize(img, tf.tensor2d(boxes), new Array(boxes.length).fill(0), [224, 224]);
        const e = net.infer(crops, true);
        return tf.div(e, tf.norm(e, 2, 1, true));
    });
    const flat = out.dataSync().slice();
    const dim = out.shape[1];
    out.dispose();
    return { flat, dim, ms: Date.now() - t0 };
}

/** What is around the medicine (pillow, quilt, table...): ImageNet top-5 for one crop. */
async function classify(msg) {
    const { w, h, data, box } = msg;
    const crop = tf.tidy(() => {
        const rgba = tf.tensor3d(new Uint8Array(data), [h, w, 4], 'int32');
        const img = tf.expandDims(tf.cast(tf.slice(rgba, [0, 0, 0], [h, w, 3]), 'float32'), 0);
        return tf.squeeze(tf.image.cropAndResize(img, tf.tensor2d([box]), [0], [224, 224]), [0]);
    });
    const top = await net.classify(crop, 5);
    crop.dispose();
    return top;
}

let heartbeat = null;

self.onmessage = async (ev) => {
    const msg = ev.data || {};
    try {
        if (msg.type === 'heartbeat') {
            // Browsers slow down timers in hidden tabs (down to once a minute); worker messages are not
            // slowed, so the page uses this tick to keep looking for medicines while you use another tab.
            if (heartbeat) clearInterval(heartbeat);
            heartbeat = msg.ms ? setInterval(() => self.postMessage({ type: 'tick' }), msg.ms) : null;
            return;
        }
        if (msg.type === 'classify') {
            if (!net) throw new Error('model not loaded');
            self.postMessage({ type: 'classes', id: msg.id, classes: await classify(msg) });
            return;
        }
        if (msg.type === 'init') {
            const backend = await init(msg);
            self.postMessage({ type: 'ready', backend });
        } else if (msg.type === 'embed') {
            if (!net) throw new Error('model not loaded');
            const r = embed(msg);
            self.postMessage({ type: 'embeds', id: msg.id, dim: r.dim, data: r.flat.buffer, ms: r.ms }, [r.flat.buffer]);
        }
    } catch (e) {
        self.postMessage({ type: 'error', id: msg.id, message: String((e && e.message) || e) });
    }
};
