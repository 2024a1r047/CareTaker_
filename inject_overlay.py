import re

with open('c:/CareTaker/frontend/voice_bot.html', 'r', encoding='utf-8') as f:
    content = f.read()

overlay = """
    <!-- Auto-Start Overlay -->
    <div id="auto-start-overlay" class="hidden fixed inset-0 bg-white/95 z-50 flex flex-col items-center justify-center cursor-pointer transition-opacity duration-300 backdrop-blur-sm" onclick="startVoiceChat()">
        <div class="bg-primary/10 p-8 rounded-full mb-6 animate-pulse">
            <i data-lucide="mic" class="w-16 h-16 text-primary"></i>
        </div>
        <h1 class="text-3xl font-bold text-slate-800 mb-2">Welcome to Saathi</h1>
        <p class="text-xl text-slate-600 mb-8 font-medium">Tap anywhere to start conversation</p>
        <button class="bg-blue-500 hover:bg-blue-600 text-white px-8 py-3 rounded-full text-lg font-semibold shadow-lg transition-all hover:scale-105">
            Start Voice Chat
        </button>
    </div>
"""

content = content.replace('<body id="body" class="bg-slate-50 min-h-screen font-sans transition-colors duration-500 relative">', '<body id="body" class="bg-slate-50 min-h-screen font-sans transition-colors duration-500 relative">\n' + overlay)

with open('c:/CareTaker/frontend/voice_bot.html', 'w', encoding='utf-8') as f:
    f.write(content)
