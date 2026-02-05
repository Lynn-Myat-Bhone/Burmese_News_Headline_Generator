import re

class MyanmarTextPreprocessor():
    def __init__(self, dict_path: str, stop_path: str):
        self.dictionary = self.load_dictionary(dict_path)
        self.stopwords = self.load_stopwords(stop_path)

    def load_dictionary(self, dict_path):
        dictionary = set()
        with open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word:
                    dictionary.add(word)
        return dictionary

    def load_stopwords(self, stopword_path):
        stopwords = set()
        with open(stopword_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word:
                    stopwords.add(word)
        return stopwords

    def merge_with_dictionary(self, syllables):
        merged_tokens = []
        i = 0
        while i < len(syllables):
            matched = False
            for j in range(len(syllables), i, -1):
                combined = ''.join(syllables[i:j])
                if combined in self.dictionary:
                    merged_tokens.append(combined)
                    i = j
                    matched = True
                    break
            if not matched:
                merged_tokens.append(syllables[i])
                i += 1
        return merged_tokens

    def preprocessing(self, text: str):
        text = re.sub(r"(([A-Za-z0-9]+)|[က-အ|ဥ|ဦ](င်္|[က-အ][ှ]*[့း]*[်]|္[က-အ]|[ါ-ှႏꩻ][ꩻ]*){0,}|.)", r"\1 ", text)
        text = text.strip().split()
        merged_tokens = self.merge_with_dictionary(text)
        return ' '.join(merged_tokens)
    

if __name__ == "__main__":
    processor = MyanmarTextPreprocessor("../dict-words.txt", "../stopwords.txt")
    text = """ကရင်ပြည်နယ် ကျားဖြန့်လုပ်ငန်းတွေကနေ ဖမ်းဆီးထိန်းသိမ်းထားတဲ့ တရုတ်နိုင်ငံသားတွေကို ပြန်ခေါ်ပေးဖို့ စစ်တပ်က တရုတ်ကိုယ်စားလှယ်ကို တောင်းဆိုခဲ့တယ်လို့ ကရင်နယ်ခြားစောင့်တပ် (BGF)ဘက်ကနေ သိရပါတယ်။ဒီဇင်ဘာ ၁၅ ရက်က မြန်မာဘက်က မြဝတီမှာ ပြုလုပ်တဲ့ ဆိုက်ဘာရာဇဝတ်မှုတိုက်ဖျက်ရေး ၃ နိုင်ငံ ဆွေးနွေးပွဲမှာ စစ်တပ်ဘက်က ပြောခဲ့တာပါ။တရုတ်နိုင်ငံရဲ့ကျားဖြန့်တိုက်ဖျက်ရေးမှာ အဓိကတာဝန်ယူထားတဲ့ ပြည်သူ့လုံခြုံရေးဝန်ကြီးဌာန လက်ထောက်ဝန်ကြီး လျိုကျုံးရီကလည်း တရုတ်နိုင်ငံသားတွေကို ပြန်လည်ခေါ်ဆောင်ဖို့ သဘောတူခဲ့တယ်လို့ BGF ဘက်ကနေ သိရပါတယ်။ထိန်းသိမ်းထားတဲ့ နိုင်ငံခြားသားတစ်ဦးအတွက် တစ်ရက်စားစရိတ် ၁၅,၀၀၀ ကျပ်နှုန်းကုန်ကျပြီး စစ်တပ်က ကုန်ကျစရိတ် ကျပ်သိန်းပေါင်း တစ်သောင်းသုံးထောင်ကျော် သုံးစွဲကုန်ကျခဲ့တယ်လို့ စစ်ကော်မရှင် ပြောခွင့်ရ ဗိုလ်ချုပ်ဇော်မင်းထွန်းက မနေ့ကပြုလုပ်တဲ့ သတင်းစာရှင်းလင်းပွဲမှာ ပြောပါတယ်။“နိုင်ငံခြားသားတွေကို သက်ဆိုင်ရာ နိုင်ငံက လာမခေါ်မချင်း လူသားချင်းစာနာထောက်ထားပြီး ထိန်းသိမ်းထားရတာ တာဝန်ကြီးလေးပါတယ်။ ကျန်ရှိနေသေးသူတွေကို သက်ဆိုင်ရာနိုင်ငံများအနေနဲ့ မိမိနိုင်ငံသားကို အမြန်ဆုံးလာရောက်ခေါ်ဆောင်ပေးကြဖို့ကိုလည်း ပြောကြားလိုပါတယ်” လို့ စစ်ကော်မရှင်ပြောခွင့်ရက ပြောထားတာပါ။စစ်ကော်မရှင်ရဲ့ အချက်အလက်အရ မြဝတီနယ်ထဲက ကျားဖြန့်ဆိုက်တွေကနေ ဒီနှစ် ဇန်နဝါရီ ၃၀ ရက်ကနေ ဒီဇင်ဘာ ၁၃ ရက်အထိ နိုင်ငံခြားသား ၁၃,၂၇၂ ဦးကို ရှာဖွေထိန်းသိမ်းထားနိုင်ပြီး ၁၁၆၁၇ ဦးကို သက်ဆိုင်ရာ နိုင်ငံအသီးသီးဆီ ပြန်လွှဲအပ်ခဲ့ပြီ ဖြစ်ပါတယ်။အများစုဟာ တရုတ်နိုင်ငံသားတွေဖြစ်ကြပါတယ်။
"""   
    print("Original:", text)

    syllables = processor.preprocessing(text)
    print("\nSyllables:")
    print(syllables)

    merged = processor.merge_with_dictionary(syllables)
    print("\nAfter dictionary merge:")
    print(merged)

    final_text = processor.preprocessing(text)
    print("\nFinal output:")
    print(final_text)