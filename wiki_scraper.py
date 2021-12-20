import wikipedia as wiki
import re




wiki.set_lang('nl')


items = wiki.random(1000)

a = wiki.page(items[0]).content
print(re.sub(r'([^A-Za-z ])+', "", a))



def preproccess_text(text):
    text = text.lower()
    stripped = re.sub(r'([^A-Za-z ])+', "", text)
    prepped = re.sub(r'([^A-Za-z])+', "_", stripped)









wiki.set_lang('fr')