import pandas as pd
import evaluate

def bleu():
    import pandas as pd
    bleu = evaluate.load("bleu")
    df = pd.read_csv("chitchat/emp_data_out.csv")
    bot_out = []
    out = []
    for index, row in df.iterrows():
        if not pd.isnull(row['bot_out']) and not pd.isnull(row['recv']):
            out.append(row['recv'].strip())
            bot_out.append(row['bot_out'].strip())
    results = bleu.compute(predictions=bot_out,references=out)
    print(results)

bleu()