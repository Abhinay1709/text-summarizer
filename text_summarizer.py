from transformers import pipeline

def main():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    print("Enter the text to summarize:")
    text = input()

    if len(text.split()) < 40:
        print("\n⚠️ Please enter at least 40 words for a good summary.")
        return

    summary = summarizer(text, max_length=120, min_length=30, do_sample=False)

    print("\n--- Summary ---")
    print(summary[0]['summary_text'])

if __name__ == "__main__":
    main()
