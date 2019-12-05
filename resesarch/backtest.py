import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def cum_pnl(weight, ret_frame):
    pnl_matrix = weight.shift(1)*ret_frame
    daily_pnl = np.nansum(pnl_matrix, axis=1)
    cum_pnl = np.cumsum(daily_pnl)
    return pd.Series(cum_pnl, index=ret_frame.index)


def turnover(weight, GMV=1.0):
    abs_weight_diff = np.abs(weight.diff())
    abs_turnover = np.sum(abs_weight_diff, axis=1)
    turnover_rate = abs_turnover/GMV
    
    start_year = weight.index[0].year
    end_year = weight.index[-1].year
    
    res = []
    for year in range(start_year, end_year+1):
        turnover_rate_this_year = turnover_rate[str(year)]
        res.append(np.nanmean(turnover_rate_this_year))
    
    return pd.Series(res, index=[str(year) for year in range(start_year, end_year+1)])


def Sharpe(weight, ret_frame, GMV=1):
    pnl_matrix = weight.shift(1)*ret_frame
    daily_pnl = np.nansum(pnl_matrix, axis=1)
    daily_return = pd.Series(daily_pnl/GMV, index=weight.index)
    start_year = weight.index[0].year
    end_year = weight.index[-1].year
    
    res = []
    for year in range(start_year, end_year+1):
        daily_return_this_year = daily_return[str(year)]
        res.append(np.nanmean(daily_return_this_year)/np.nanstd(daily_return_this_year)*np.sqrt(len(daily_return_this_year)))
    
    return pd.Series(res, index=[str(year) for year in range(start_year, end_year+1)])


def MDD(weight, ret_frame, GMV=1.0):
    pnl = cum_pnl(weight, ret_frame)
    start_year = weight.index[0].year
    end_year = weight.index[-1].year
    
    MDD_yearly = pd.Series(0.0, index=[str(year) for year in range(start_year, end_year+1)])
    for year in range(start_year, end_year+1):
        pnl_this_year = pnl[str(year)]
        MDD_this_year = pd.Series(np.nan, index=pnl_this_year.index)
        
        max_pnl = -np.inf
        cur_mdd = 0
        
        for i in range(len(pnl_this_year)):
            if pnl_this_year[i] > max_pnl:
                max_pnl = pnl_this_year[i]
            elif (pnl_this_year[i]-max_pnl)/GMV < cur_mdd:
                cur_mdd = (pnl_this_year[i]-max_pnl)/GMV
            MDD_this_year[i] = cur_mdd
        
        MDD_yearly[str(year)] = MDD_this_year[-1]
    
    return MDD_yearly


def annual_return(weight, ret_frame, GMV=1.0):
    pnl = cum_pnl(weight, ret_frame)
    
    start_year = weight.index[0].year
    end_year = weight.index[-1].year
    
    res = []
    for year in range(start_year, end_year+1):
        pnl_this_year = pnl[str(year)]
        res.append((pnl_this_year[-1]-pnl_this_year[0])/GMV)
    return pd.Series(res, index=[str(year) for year in range(start_year, end_year+1)])


def display_backtest(weight, ret_frame, GMV=1.0, show_plot=True):
    pnl = cum_pnl(weight, ret_frame)
    if show_plot:
        plt.figure(figsize=(10, 5))
        plt.plot(pnl)
        plt.xlabel('time')
        plt.ylabel('pnl')
        plt.title('pnl plot of strategy')
        plt.show()
    
    turnover_res = turnover(weight, GMV)
    Sharpe_res = Sharpe(weight, ret_frame, GMV)
    MDD_res = MDD(weight, ret_frame, GMV)
    annual_return_res = annual_return(weight, ret_frame, GMV)
    
    backtest_res = pd.concat([turnover_res, Sharpe_res, MDD_res, annual_return_res], axis=1)
    backtest_res.columns = ['turnover', 'Sharpe ratio', 'maximum drawdown', 'annual return']
    backtest_res.loc['average'] = np.nanmean(backtest_res, axis=0)
    backtest_res['maximum drawdown'][-1] = np.nanmin(backtest_res['maximum drawdown'][:-1])
    
    if show_plot:
        display(backtest_res)
    
    return pnl, backtest_res


