import pandas as pd


class ReturnSeries:
    """
    Given a csv file path return a return series.
    
    path - a string file path
    name - the name of the output col
    col_key - the name of the input col
    is_percent - if the values are in percentage format
    """
    def __init__(self, path, name, col_key, is_percent=True):
        self.path = path
        self.name = name
        self.col_key = col_key
        df = pd.read_csv(path, parse_dates=True)
        df.index = df.index.to_period('M')
        df = df[col_key]
        df.columns = [name]
        df = df.pct_change()
        df = df.dropna()
        if isPercent:
            self.df = self.df / 100
        self.returns = df
        
                
                              
    
    