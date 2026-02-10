from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_dir = "model_v2"
model = AutoModelForSeq2SeqLM.from_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained(output_dir)
print("Model loaded successfully!")

def generate_headline(article_text, max_length=64, num_beams=4, temperature=0.7):
    """Generate headline with better parameters"""
    input_text = "summarize: " + article_text
    inputs = tokenizer(
        input_text,
        max_length=max_length,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=2,
            length_penalty=1.0,
            temperature=temperature
        )

    headline = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return headline




custom_article = """
အာရက္ခတပ်တော်(အေအေ) စစ်ဦးစီးချုပ်ထွန်းမြတ်နိုင်နဲ့ မရာနယ်မြေအုပ်ချုပ်ကောင်စီတို့ တွေ့ဆုံခဲ့တယ်ဆိုပြီး ဒီကနေ့ ဖေဖော်ဝါရီ ၁၀ ရက်မှာ မရာနယ်မြေအုပ်ချုပ်ရေးကောင်စီ/မရာလဲန်းကာကွယ်ရေးတပ်ဖွဲ့(MTC/MDF)က ထုတ်ပြန်ပါတယ်။အနာဂတ်မှာလည်း မိတ်ဖက်စိတ်ဓာတ်၊ ယုံကြည်မှုနဲ့ နားလည်မှုများအပေါ် အခြေခံကာ ပိုမိုခိုင်မာတဲ့ လက်တွဲပူးပေါင်းမှုများ တည်ဆောက်သွားလိုတယ်လို့ ပြောထားပါတယ်။အဲဒီတွေ့ဆုံမှုကို ၂၀၂၅ ခုနှစ် စက်တင်ဘာ ၉ ရက်နေ့ကတည်းက ပြုလုပ်ထားတာလို့ MTC ရဲ့ ထုတ်ပြန်ချက်အရ နားလည်ရပါတယ်။မရာလူမျိုးတွေက ချင်းတိုင်းရင်းသားထဲက လူမျိုးစု တစ်စုဖြစ်ပြီး ပလက်ဝ၊ မတူပီနဲ့ မင်းတပ်တို့မှာ အခြေချနေထိုင်မှုများတဲ့ လူမျိုးစုဖြစ်ပါတယ်။
"""

print("Custom Article:")
print(custom_article)
print("\nGenerated Headline:")
print(generate_headline(custom_article.strip()))