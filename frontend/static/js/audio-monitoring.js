// Audio Monitoring Module
document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const startListeningBtn = document.getElementById('startListening');
    const audioStatus = document.getElementById('audioStatus');
    const audioLevelMeter = document.getElementById('audioLevelMeter');
    const detectedEventList = document.getElementById('detectedEventList');
    const waveCanvas = document.getElementById('waveCanvas');
    let waveCtx = waveCanvas ? waveCanvas.getContext('2d') : null;
    
    // Audio context and variables
    let audioContext;
    let analyser;
    let microphone;
    let isListening = false;
    let isVisualizing = false;
    let model;
    let dataArray;
    let animationId;
    
    // Audio detection configuration
    const DETECTION_THRESHOLD = 0.78; // 78% confidence threshold
    const ALERT_THRESHOLD = 0.90;     // 90% confidence for alerts

    // Detection state
    let detectionActive = {
        cough: false,
        snore: false
    };
    let detectionStartTime = {
        cough: null,
        snore: null
    };
    let detectionTimeouts = {
        cough: null,
        snore: null
    };
    let detectionCounts = {
        cough: 0,
        snore: 0
    };
    
    // Track total detections for percentage calculation
    let totalDetections = 0;
    let coughDetections = 0;
    
    // Initialize audio monitoring
    async function initAudioMonitoring() {
        try {
            // Load the TensorFlow.js speech commands model
            model = await window.speechCommands.create('BROWSER_FFT');
            await model.ensureModelLoaded();
            
            // Get the class labels from the model
            const labels = model.wordLabels();
            console.log('Model loaded with labels:', labels);
            
            // Set up audio context
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            analyser = audioContext.createAnalyser();
            analyser.fftSize = 2048;
            const bufferLength = analyser.frequencyBinCount;
            dataArray = new Uint8Array(bufferLength);
            
            // Set up the audio processing
            setupAudioProcessing();
            
            // Start visualization if we have a canvas
            if (waveCanvas) {
                setupCanvas();
                startVisualization();
            }
            
            // Update UI
            audioStatus.textContent = 'Ready to monitor';
            startListeningBtn.disabled = false;
            
        } catch (error) {
            console.error('Error initializing audio monitoring:', error);
            audioStatus.textContent = 'Error initializing audio';
        }
    }
    
    // Set up audio processing
    function setupAudioProcessing() {
        // Get user media (microphone access)
        navigator.mediaDevices.getUserMedia({ audio: true, video: false })
            .then(function(stream) {
                // Create a media stream source
                microphone = audioContext.createMediaStreamSource(stream);
                
                // Connect the microphone to the analyser
                microphone.connect(analyser);
                
                // Set up the analyser
            analyser.fftSize = 2048;
            const bufferLength = analyser.frequencyBinCount;
            dataArray = new Uint8Array(bufferLength);
            
            // Connect the microphone to the analyser and destination
            microphone.connect(analyser);
            
            // Start the analysis loop
            function analyze() {
                if (!isListening) return;
                
                // Get the time domain data for the waveform
                analyser.getByteTimeDomainData(dataArray);
                
                // Calculate the average volume (for the volume meter)
                let sum = 0;
                for (let i = 0; i < bufferLength; i++) {
                    sum += Math.abs(dataArray[i] - 128);
                }
                const average = sum / bufferLength;
                
                // Update the volume meter
                const normalizedVolume = average / 128; // 128 is the middle value (0-255)
                updateVolumeMeter(normalizedVolume);
                
                // Continue the analysis loop
                animationId = requestAnimationFrame(analyze);
            }
                
                // Start the analysis
                if (animationId) {
                    cancelAnimationFrame(animationId);
                }
                analyze();
                
            })
            .catch(function(err) {
                console.error('Error accessing microphone:', err);
                audioStatus.textContent = 'Microphone access denied';
            });
    }
    
    // Update the volume meter
    function updateVolumeMeter(level) {
        const percentage = Math.min(100, Math.max(0, Math.round(level * 100)));
        audioLevelMeter.style.width = `${percentage}%`;
        
        // Change color based on level
        if (level > 0.7) {
            audioLevelMeter.style.backgroundColor = '#C96B55'; // Red for loud sounds
        } else if (level > 0.4) {
            audioLevelMeter.style.backgroundColor = '#E8B89A'; // Orange for medium sounds
        } else {
            audioLevelMeter.style.backgroundColor = '#91AA8A'; // Green for quiet sounds
        }
    }
    
    // Toggle audio monitoring
    async function toggleAudioMonitoring() {
        if (isListening) {
            // Stop listening
            if (microphone) {
                microphone.disconnect();
            }
            if (audioContext && audioContext.state !== 'closed') {
                await audioContext.close();
            }
            stopVisualization();
            
            isListening = false;
            startListeningBtn.textContent = 'Start Listening';
            audioStatus.textContent = 'Monitoring stopped';
            
        } else {
            // Start listening
            try {
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                analyser = audioContext.createAnalyser();
                
                // Get microphone access
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                microphone = audioContext.createMediaStreamSource(stream);
                microphone.connect(analyser);
                
                // Start the detection
                isListening = true;
                startListeningBtn.textContent = 'Stop Listening';
                audioStatus.textContent = 'Listening for sounds...';
                
                // Start the analysis loop
                function analyze() {
                    if (!isListening) return;
                    
                    // Get the frequency data
                    const bufferLength = analyser.frequencyBinCount;
                    const dataArray = new Uint8Array(bufferLength);
                    analyser.getByteFrequencyData(dataArray);
                    
                    // Calculate the average volume
                    let sum = 0;
                    for (let i = 0; i < bufferLength; i++) {
                        sum += dataArray[i];
                    }
                    const average = sum / bufferLength;
                    
                    // Update the volume meter
                    const normalizedVolume = average / 255;
                    updateVolumeMeter(normalizedVolume);
                    
                    // Continue the analysis loop
                    requestAnimationFrame(analyze);
                }
                
                // Start the analysis
                if (animationId) {
                    cancelAnimationFrame(animationId);
                }
                analyze();
                
                // Start the speech recognition
                await model.listen(result => {
                    // Get the scores for each class
                    const scores = Array.from(result.scores);
                    const labels = model.wordLabels();
                    
                    // Create an object to store scores by label
                    const scoresByLabel = {};
                    labels.forEach((label, index) => {
                        scoresByLabel[label.toLowerCase()] = scores[index];
                    });
                    
                    // Log scores for debugging
                    console.log('Scores by label:', scoresByLabel);
                    
                    // Update UI with scores
                    updateScoresUI(
                        scoresByLabel['background noise'] || 0,
                        scoresByLabel['cough'] || 0,
                        scoresByLabel['snore'] || 0
                    );
                    
                    // Handle detections for each class
                    if (scoresByLabel['cough'] >= DETECTION_THRESHOLD) {
                        handleDetection('cough', scoresByLabel['cough']);
                    }
                    if (scoresByLabel['snore'] >= DETECTION_THRESHOLD) {
                        handleDetection('snore', scoresByLabel['snore']);
                    }
                    
                }, {
                    includeSpectrogram: true,
                    probabilityThreshold: 0.7,
                    invokeCallbackOnNoiseAndUnknown: false,
                    overlapFactor: 0.5
                });
                
            } catch (error) {
                console.error('Error starting audio monitoring:', error);
                audioStatus.textContent = 'Error: ' + error.message;
                isListening = false;
                startListeningBtn.textContent = 'Start Listening';
            }
        }
    }
    
    // Handle detection events (cough or snore)
    function handleDetection(type, score) {
        const currentTime = Date.now();
        const confidencePercent = Math.round(score * 100);
        
        // Increment total detections for percentage calculation
        totalDetections++;
        
        // Track cough detections separately for percentage
        if (type === 'cough' && score >= DETECTION_THRESHOLD) {
            coughDetections++;
        }
        
        // Show browser alert for high confidence detections
        if (score >= ALERT_THRESHOLD) {
            if (!detectionActive[type] || (currentTime - detectionStartTime[type]) > 1000) {
                detectionActive[type] = true;
                detectionStartTime[type] = currentTime;
                
                // Show alert
                alert(`🚨 ${type.toUpperCase()} DETECTED (${confidencePercent}% confidence)`);
                
                // Add to detected events
                addDetectedEvent(type, score);
                
                // Start recording and update counts
                detectionCounts[type]++;
                saveDetectionRecording(type, score);
                
                // Update the UI with new counts and percentage
                const scores = {
                    'background noise': document.getElementById('backgroundScore')?.textContent || 0,
                    'cough': document.getElementById('coughScore')?.textContent || 0,
                    'snore': document.getElementById('snoreScore')?.textContent || 0
                };
                updateScoresUI(scores['background noise'], scores['cough'], scores['snore']);
            }
        }
        
        // Reset detection state after a delay
        if (detectionTimeouts[type]) {
            clearTimeout(detectionTimeouts[type]);
        }
        detectionTimeouts[type] = setTimeout(() => {
            detectionActive[type] = false;
            detectionStartTime[type] = null;
        }, 2000);
    }
    
    // Save detection recording
    async function saveDetectionRecording(type, score) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const mediaRecorder = new MediaRecorder(stream);
            const audioChunks = [];
            
            mediaRecorder.ondataavailable = (event) => {
                audioChunks.push(event.data);
            };
            
            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                const formData = new FormData();
                formData.append('audio', audioBlob, `${type}_${detectionCounts[type]}_${Date.now()}.wav`);
                formData.append('confidence', score.toString());
                formData.append('type', type);
                
                try {
                    const response = await fetch('/api/recordings', {
                        method: 'POST',
                        headers: {
                            'Authorization': `Bearer ${localStorage.getItem('token')}`
                        },
                        body: formData
                    });
                    
                    if (!response.ok) {
                        console.error(`Failed to save ${type} recording`);
                    }
                } catch (error) {
                    console.error(`Error saving ${type} recording:`, error);
                }
            };
            
            mediaRecorder.start();
            setTimeout(() => {
                mediaRecorder.stop();
                stream.getTracks().forEach(track => track.stop());
            }, 3000); // Record for 3 seconds
            
        } catch (error) {
            console.error(`Error accessing microphone for ${type} recording:`, error);
        }
    }
    
    // Update UI with detection scores
    function updateScoresUI(background, cough, snore) {
        // Update score displays
        updateScore('backgroundScore', background);
        updateScore('coughScore', cough);
        updateScore('snoreScore', snore);
        
        // Update meters
        updateMeter('backgroundMeter', background);
        updateMeter('coughMeter', cough);
        updateMeter('snoreMeter', snore);
        
        // Update cough percentage if we have detections
        if (totalDetections > 0) {
            const coughPercentage = Math.round((coughDetections / totalDetections) * 100);
            const coughPercentageElement = document.getElementById('coughPercentage');
            if (coughPercentageElement) {
                coughPercentageElement.textContent = `${coughPercentage}%`;
            }
        }
        
        // Update total cough count
        const coughCountElement = document.getElementById('coughCount');
        if (coughCountElement) {
            coughCountElement.textContent = detectionCounts.cough;
        }
    }
    
    // Update a score display
    function updateScore(elementId, value) {
        const element = document.getElementById(elementId);
        if (element) {
            element.textContent = (value * 100).toFixed(1) + '%';
        }
    }
    
    // Update a meter display
    function updateMeter(elementId, value) {
        const meter = document.getElementById(elementId);
        if (meter) {
            meter.style.width = `${value * 100}%`;
            
            // Set color based on threshold
            if (value >= DETECTION_THRESHOLD) {
                meter.style.backgroundColor = '#C96B55'; // Red for detection
            } else if (value >= (DETECTION_THRESHOLD * 0.7)) {
                meter.style.backgroundColor = '#E8B89A'; // Orange for near detection
            } else {
                meter.style.backgroundColor = '#91AA8A'; // Green for normal
            }
        }
    }
    
    // Add a detected event to the list
    function addDetectedEvent(label, confidence) {
        const now = new Date();
        const timeString = now.toLocaleTimeString();
        const confidencePercent = (confidence * 100).toFixed(1);
        
        // Create the event element
        const eventElement = document.createElement('div');
        eventElement.className = 'detected-event';
        eventElement.innerHTML = `
            <div class="event-time">${timeString}</div>
            <div class="event-type">${label}</div>
            <div class="event-confidence">${confidencePercent}%</div>
        `;
        
        // Add to the top of the list
        if (detectedEventList.firstChild) {
            detectedEventList.insertBefore(eventElement, detectedEventList.firstChild);
        } else {
            detectedEventList.appendChild(eventElement);
        }
        
        // Keep only the last 10 events
        while (detectedEventList.children.length > 10) {
            detectedEventList.removeChild(detectedEventList.lastChild);
        }
    }
    
    // Waveform visualization functions
    function setupCanvas() {
        if (!waveCanvas) return;
        
        // Set canvas size
        waveCanvas.width = waveCanvas.offsetWidth * window.devicePixelRatio;
        waveCanvas.height = waveCanvas.offsetHeight * window.devicePixelRatio;
        waveCtx = waveCanvas.getContext('2d');
        waveCtx.scale(window.devicePixelRatio, window.devicePixelRatio);
    }
    
    function drawWaveform() {
        if (!isVisualizing || !waveCtx || !dataArray) {
            return;
        }
        
        const width = waveCanvas.width / window.devicePixelRatio;
        const height = waveCanvas.height / window.devicePixelRatio;
        
        // Clear the canvas
        waveCtx.clearRect(0, 0, width, height);
        
        // Draw background
        waveCtx.fillStyle = 'rgba(15, 23, 42, 0.5)';
        waveCtx.fillRect(0, 0, width, height);
        
        // Draw waveform
        waveCtx.lineWidth = 2;
        waveCtx.strokeStyle = '#38bdf8';
        waveCtx.beginPath();
        
        const sliceWidth = width * 1.0 / dataArray.length;
        let x = 0;
        
        for (let i = 0; i < dataArray.length; i++) {
            const v = dataArray[i] / 128.0;  // Convert to -1 to 1
            const y = v * height / 2 + height / 2;
            
            if (i === 0) {
                waveCtx.moveTo(x, y);
            } else {
                waveCtx.lineTo(x, y);
            }
            
            x += sliceWidth;
        }
        
        waveCtx.stroke();
        
        // Continue the animation
        if (isVisualizing) {
            requestAnimationFrame(drawWaveform);
        }
    }
    
    function startVisualization() {
        if (!waveCanvas) return;
        isVisualizing = true;
        drawWaveform();
    }
    
    function stopVisualization() {
        isVisualizing = false;
        if (animationId) {
            cancelAnimationFrame(animationId);
            animationId = null;
        }
    }
    
    // Handle window resize
    window.addEventListener('resize', () => {
        if (waveCanvas) {
            setupCanvas();
        }
    });
    
    // Initialize the audio monitoring when the page loads
    if (startListeningBtn) {
        startListeningBtn.addEventListener('click', toggleAudioMonitoring);
        initAudioMonitoring();
    }
});
