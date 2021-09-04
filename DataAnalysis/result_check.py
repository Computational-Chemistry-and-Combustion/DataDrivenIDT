'''
This files Generated the pairwise plot for checking the distribution against
other parameters
'''
import pandas as pd 
import numpy as np
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
from search_fileNcreate import search_fileNcreate as SF

class result_check:
    def pairwise_plot(data):
        '''
        This method will generate pairwise plot
        Input : 
        data - dataframe
        '''
        ###Pairwise plot
        sns.pairplot(data.dropna(), vars = list(data.columns) )
        plt.show()

    def random_color():
        rgbl=[255,0,0]
        random.shuffle(rgbl)
        return tuple(rgbl)
    
    def VIF(df,curr_directory,child_type='root',cluster = 0):
        '''
        VIF : variation Inflation factor 
        '''
        print('''
        \n The variance inflation factor (VIF) to check the multi-collinearity of the features 
        \n Value = 1 means feature has no multi-collinearity 
        \n value = 1-5 means feature has moderate correlation 
        \n value > 5  it will poorly identify the coefficient  
        ''')
        # For each X, calculate VIF and save in dataframe
        vif = pd.DataFrame()

        # vif["features"] = df.columns
        # try:
        features = list(df.columns)
        del features[0] #removing first feature constant 
        vif["features"] = features
        vif["VIF Factor"] = [variance_inflation_factor(df.values, i) for i in range(1,df.shape[1])]
        print(vif.round(1))
        SF.check_directory(str(curr_directory)+'/result/vif/'+str(child_type)) #checking directory
        vif.to_csv(str(curr_directory)+'/result/vif/'+str(child_type)+'/vif_'+str(cluster)+'.csv',index=False)
        print("\nPlease check ./result/vif/"+str(child_type)+"/vif_"+str(cluster)+".csv")
        # except AttributeError or  IndexError :
        #     pass
	    

    def results_print(model):
        '''
        This method will generate all the summary parameter
        input: model (OBJECT of regressor)
        '''
        # print('Summary:',model.summary())
        # print('params:',model.params)
        # print('tvalues:',model.tvalues)
        # print('ttest:',model.t_test([1, 0]))
        # print('ftest:',model.f_test(np.identity(2)))
        print('Condition Number:',model.condition_number)
        # print('Eigen values:',model.eigenvals)
        print('R squared:',model.rsquared)
        print('F-statistic:', model.fvalue)
        print('Prob (F-statistic):', model.f_pvalue)
        print('Log-Likelihood:', None)
        print('AIC:', model.aic)
        print('BIC:', model.bic)
        print('SSR:',model.ssr)
        # print('Confidence intervasl:',model.conf_int())
        print('Number of observations :',model.nobs)
        # print('Predicted Values:',model.fittedvalues)
        # print('Residual:',model.resid)
        print('Scale:',model.scale)
        print('Centered_tss:',model.centered_tss)
        print('ESS:',model.ess)
        print('R squared_adj:',model.rsquared_adj)
        print('MSE Model:',model.mse_model)
        print('MSE_residuals:',model.mse_resid)
        print('MSE Total:',model.mse_total)
        # print(' Residuals Pearson:',model.resid_pearson)
        print('Residual Transformed:',)
        from statsmodels.stats.stattools import (jarque_bera, omni_normtest, durbin_watson)
        wresid_result = model.wresid
        jb, jbpv, skew, kurtosis = jarque_bera(wresid_result)
        omni, omnipv = omni_normtest(wresid_result)
        dw = durbin_watson(wresid_result)
        print('Omnibus:',omni)
        print('Prob(Omnibus):',omnipv)
        print('Skewnes:',skew)
        print('Kurtosis:',kurtosis)
        print('Jarque-Bera:',jb)
        print('JB Probability:',jbpv)
        print('Durbin-Watson:',dw)