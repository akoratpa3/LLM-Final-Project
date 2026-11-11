Requirements!


Use transformers==4.55.0. With the newer 5.x, the model loads and writes fluent text with no errors, but accuracy drops from ~94% to ~33%. Restart the session after installing.
Use the authors’ prompt format. Build prompts with compose_text() from utils/text_preprocess.py: "<question> {problem} {question} </question> <solution>" (single spaces, no newline).
Evaluate in float32 (or float16, the authors’ setting). bfloat16 hurt accuracy in our early tests.
