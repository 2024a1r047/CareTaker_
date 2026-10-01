import sys
sys.path.append('c:/CareTaker/backend')
from services.voice_bot_engine import generate_system_prompt
import re

p = generate_system_prompt('Test', {}, 'hi', {})
p_over = re.sub(
    r'For music/story: set "recommendation".*?"search"\.',
    'REPLACED',
    p, flags=re.DOTALL
)

if 'REPLACED' in p_over:
    print("SUCCESS")
else:
    print("FAILED")
    print(p)
