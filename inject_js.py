import re

with open('c:/CareTaker/frontend/voice_bot.html', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update labels
old_labels_regex = r"const labels = \{[\s\S]*?talking: 'Speaking.*?'\n\s*\};"
new_labels = """const labels = {
                idle: 'Saathi • Ready',
                listening: 'Listening...',
                thinking: 'Replying...',
                talking: 'Speaking...'
            };"""
content = re.sub(old_labels_regex, new_labels, content)

# 2. Add startVoiceChat() function before init()
if 'function startVoiceChat()' not in content:
    start_func = """
        function startVoiceChat() {
            const overlay = document.getElementById('auto-start-overlay');
            if (overlay) overlay.classList.add('hidden');
            if (!state.isListening) {
                toggleListening();
            }
        }

        async function init() {"""
    content = content.replace("async function init() {", start_func)

# 3. Add overlay reveal logic inside init()
if 'auto-start-overlay' not in content.split('requestNotificationPermission();')[1]:
    init_logic = """requestNotificationPermission();
            
            // Show auto-start overlay on load
            if (!state.isListening) {
                const overlay = document.getElementById('auto-start-overlay');
                if (overlay) overlay.classList.remove('hidden');
            }"""
    content = content.replace("requestNotificationPermission();", init_logic)

with open('c:/CareTaker/frontend/voice_bot.html', 'w', encoding='utf-8') as f:
    f.write(content)
