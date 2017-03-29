#!/usr/bin/env python3
"""
Main script. See README.md for more information

Use python 3
"""

import chatbot
import sys


if __name__ == "__main__":
    cb = chatbot.Chatbot()
    cb.set_up_things()
    quest = ' '.join(sys.argv[1].split('_'))
    ans = cb.get_answer(str(quest))
    print(ans)
