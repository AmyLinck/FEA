import pandas as pd
from scipy import stats

if __name__ == "__main__":
    sheet = 'f5'
    headers = [' DG', ' Single', ' ODG', " Tree", " Tree2", ' PSO']
    df = pd.read_excel('./results.xlsx', sheet_name=sheet)
    for col in df.columns:
        print(col)
    # exit()
    for x_head in headers:
        for y_head in headers:
            if x_head == y_head:
                continue
            x = df.loc[:, x_head].to_numpy()[:25]
            y = df.loc[:, y_head].to_numpy()[:25]

            print(f'{x_head} vs {y_head} = {stats.ranksums(x, y)}')

