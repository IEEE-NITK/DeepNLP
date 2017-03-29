import chatbot

cb = chatbot.Chatbot()
cb.set_up_things()

def q2a_generative(query):
  return cb.get_answer(str(query))

if __name__ == "__main__":
  while True:
    quest = input("Q> ")
    ans = cb.get_answer(str(quest))
    print("A> " + ans)
