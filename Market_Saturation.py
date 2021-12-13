import pandas as pd
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm

##Import data
raw=r'C:\Users\etfisher\Desktop\Python Final\market_saturation_and_utilization_cbsa_dataset_2021-04-16 (3).csv'
cbsa=pd.read_csv(raw)
raw2=r'C:\Users\etfisher\Desktop\Python Final\cbsa-est2019-alldata.csv'
pop=pd.read_csv(raw2)

##Rename columns
cbsa.columns=['year', 'type_of_service', 'aggregation_level','cbsa',
              'cbsa_name','num_of_fee_for_service__beneficiaries',
              'num_of_providers','avg_num_of_users_per_provider',
              'percentage_of_users_out_of_ffs_beneficiaries','num_of_users',
              'avg_num_providers_per_cbsa','num_dual_eligible_users',
              'percent_dual_eligible_users_out_of_total',
              'percent_dual_eligible_users_out_of_dual_eligible_ffs_benificiaries',
              'total_payment','percentage_of_ffs_beneficiaries_by_num_of_providers_serving_cbsa_0to2',
              'percentage_of_ffs_beneficiaries_by_num_of_providers_serving_cbsa_3to4',
              'percentage_of_ffs_beneficiaries_by_num_of_providers_serving_cbsa_5to9',
              'percentage_of_ffs_beneficiaries_by_num_of_providers_serving_cbsa_10to19',
              'percentage_of_ffs_beneficiaries_by_num_of_providers_serving_cbsa_20plus']
pop.columns=['cbsa', 'mdiv', 'stcou', 'cbsa_name', 'lsad', 'census2010pop', 
             'estimates_base2010', 'pop_estimate2010', 'pop_estimate2011', 
             'pop_estimate2012', 'pop_estimate2013', 'pop_estimate2014', 
             'pop_estimate2015', 'pop_estimate2016', 'pop_estimate2017', 
             'pop_estimate2018', 'pop_estimate2019', 'npop_change2010', 
             'npop_change2011', 'npop_change2012', 'npop_change2013', 
             'npop_change2014', 'npop_change2015', 'npop_change2016',
             'npop_change2017', 'npop_change2018', 'npop_change2019',
             'births2010', 'births2011', 'births2012', 'births2013',
             'births2014', 'births2015', 'births2016', 'births2017', 
             'births2018', 'births2019', 'deaths2010', 'deaths2011', 
             'deaths2012', 'deaths2013', 'deaths2014', 'deaths2015', 
             'deaths2016', 'deaths2017', 'deaths2018', 'deaths2019',
             'natrual_inc2010', 'natrual_inc2011', 'natrual_inc2012',
             'natrual_inc2013', 'natrual_inc2014', 'natrual_inc2015', 
             'natrual_inc2016', 'natrual_inc2017', 'natrual_inc2018',
             'natrual_inc2019', 'international_mig2010', 'international_mig2011',
             'international_mig2012', 'international_mig2013',
             'international_mig2014', 'international_mig2015', 
             'international_mig2016', 'international_mig2017', 
             'international_mig2018', 'international_mig2019', 
             'domestic_mig2010', 'domestic_mig2011', 'domestic_mig2012', 
             'domestic_mig2013', 'domestic_mig2014', 'domestic_mig2015',
             'domestic_mig2016', 'domestic_mig2017', 'domestic_mig2018',
             'domestic_mig2019', 'net_mig2010', 'net_mig2011', 'net_mig2012',
             'net_mig2013', 'net_mig2014', 'net_mig2015', 'net_mig2016',
             'net_mig2017', 'net_mig2018', 'net_mig2019', 'residual2010',
             'residual2011', 'residual2012', 'residual2013', 'residual2014',
             'residual2015', 'residual2016', 'residual2017', 'residual2018',
             'residual2019']

##Drop unneccissary columns
cbsa.drop(['avg_num_of_users_per_provider', 'percentage_of_users_out_of_ffs_beneficiaries',
           'avg_num_providers_per_cbsa', 'percent_dual_eligible_users_out_of_total',
           'percent_dual_eligible_users_out_of_dual_eligible_ffs_benificiaries',
           'percentage_of_ffs_beneficiaries_by_num_of_providers_serving_cbsa_0to2',
           'percentage_of_ffs_beneficiaries_by_num_of_providers_serving_cbsa_3to4',
           'percentage_of_ffs_beneficiaries_by_num_of_providers_serving_cbsa_5to9',
           'percentage_of_ffs_beneficiaries_by_num_of_providers_serving_cbsa_10to19',
           'percentage_of_ffs_beneficiaries_by_num_of_providers_serving_cbsa_20plus'],
          axis=1, inplace=True)
pop.drop(['mdiv', 'census2010pop','estimates_base2010', 'pop_estimate2010','pop_estimate2011', 'pop_estimate2012', 'pop_estimate2013','pop_estimate2014', 'npop_change2010', 'npop_change2011', 'npop_change2012', 'npop_change2013', 'npop_change2014', 'npop_change2015', 'npop_change2016','npop_change2017', 'npop_change2018', 'npop_change2019','births2010', 'births2011', 'births2012', 'births2013','births2014', 'births2015', 'births2016', 'births2017', 'births2018', 'births2019', 'deaths2010', 'deaths2011', 'deaths2012', 'deaths2013', 'deaths2014', 'deaths2015', 'deaths2016', 'deaths2017', 'deaths2018', 'deaths2019','natrual_inc2010', 'natrual_inc2011', 'natrual_inc2012','natrual_inc2013', 'natrual_inc2014', 'natrual_inc2015', 'natrual_inc2016', 'natrual_inc2017', 'natrual_inc2018',
          'natrual_inc2019', 'international_mig2010', 'international_mig2011',
          'international_mig2012', 'international_mig2013',
          'international_mig2014', 'international_mig2015', 
          'international_mig2016', 'international_mig2017', 
          'international_mig2018', 'international_mig2019', 
          'domestic_mig2010', 'domestic_mig2011', 'domestic_mig2012', 
          'domestic_mig2013', 'domestic_mig2014', 'domestic_mig2015',
          'domestic_mig2016', 'domestic_mig2017', 'domestic_mig2018',
          'domestic_mig2019', 'net_mig2010', 'net_mig2011', 'net_mig2012',
          'net_mig2013', 'net_mig2014', 'net_mig2015', 'net_mig2016',
          'net_mig2017', 'net_mig2018', 'net_mig2019', 'residual2010',
          'residual2011', 'residual2012', 'residual2013', 'residual2014',
          'residual2015', 'residual2016', 'residual2017', 'residual2018',
          'residual2019'],axis=1, inplace=True)

##Melt 'pop' data
pop_long=pd.melt(pop,id_vars=['cbsa', 'stcou', 'cbsa_name', 'lsad'], 
                 var_name='year', value_name='population')

##Re-word columns
cbsa['year']=cbsa['year'].str[:4]
pop_long['year']=pop_long['year'].str[12:]

##Change data type
data_change={'cbsa': str}
cbsa=cbsa.astype(data_change)
pop_long=pop_long.astype(data_change)

##Merge Data
df=cbsa.merge(pop_long, how='inner', left_on=['cbsa','year'], 
              right_on=['cbsa','year'])

##Count unique values in the 'type_of_service' column
unique_services=df.type_of_service.value_counts()

##Check for missing values in 'df'
df.isnull().sum()

##Replace missing 'num_dual_eligible_users' values with '0'
##Replace missing 'STCOU' values with '0'
df['num_dual_eligible_users'].fillna(value=0, inplace=True)
df['stcou'].fillna(value=0, inplace=True)

##Drop unneccissary rows
df=df[df.stcou == 0]
df=df[df.num_of_providers > 0]

##Remove 'stcou'
df.drop(['stcou'],axis=1, inplace=True)

##View correlations
df_corr=df.corr()

##Create new data set that is sorted by 'type of service' and 'num_of_providers'
df_services_providers=df.groupby(['type_of_service','num_of_providers']).sum()
df_services_providers=df_services_providers.reset_index()

##Create new data set that is sorted by 'type_of_service' and 'num_of_users'
df_services_users=df.groupby(['type_of_service','num_of_users']).sum()
df_services_users=df_services_users.reset_index()

##Create new data set that is sorted by 'type_of_service' and 'total_payment'
df_services_payments=df.groupby(['type_of_service','total_payment']).mean()
df_services_payments=df_services_payments.reset_index()

##New datasets for linear regressions in scatter plot
##Create new dataset that only includes Ambulance (Emergency & Non-Emergency)
df_amb1=df[df.type_of_service == 'Ambulance (Emergency & Non-Emergency)']
##Create new dataset that only includes Ambulance (Emergency)
df_amb2=df[df.type_of_service == 'Ambulance (Emergency)']
##Create new dataset that only includes Ambulance (Non-Emergency)
df_amb3=df[df.type_of_service == 'Ambulance (Non-Emergency)']
##Create new dataset that only includes Cardiac Rehabilitation Program
df_card=df[df.type_of_service == 'Cardiac Rehabilitation Program']
##Create new dataset that only includes Chiropractic Services
df_chiro=df[df.type_of_service == 'Chiropractic Services']
##Create new dataset that only includes Clinical Laboratory (Billing Independently)
df_lab=df[df.type_of_service == 'Clinical Laboratory (Billing Independently)']
##Create new dataset that only includes Dialysis
df_dialysis=df[df.type_of_service == 'Dialysis']
##Create new dataset that only includes Federally Qualified Health Center (FQHC)
df_fqhc=df[df.type_of_service == 'Federally Qualified Health Center (FQHC)']
##Create new dataset that only includes Home Health
df_home=df[df.type_of_service == 'Home Health']
##Create new dataset that only includes Hospice
df_hospice=df[df.type_of_service == 'Hospice']
##Create new dataset that only includes Independent Diagnostic Testing Facility Pt A
df_testA=df[df.type_of_service == 'Independent Diagnostic Testing Facility Pt A']
##Create new dataset that only includes Independent Diagnostic Testing Facility Pt B
df_testB=df[df.type_of_service == 'Independent Diagnostic Testing Facility Pt B']
##Create new dataset that only includes Long-Term Care Hospitals
df_ltc=df[df.type_of_service == 'Long-Term Care Hospitals']
##Create new dataset that only includes Ophthalmology
df_ophtho=df[df.type_of_service == 'Ophthalmology']
##Create new dataset that only includes Physical & Occupational Therapy
df_ptot=df[df.type_of_service == 'Physical & Occupational Therapy']
##Create new dataset that only includes Preventative Health Services
df_prevent=df[df.type_of_service == 'Preventive Health Services']
##Create new dataset that only includes Psychotherapy
df_psych=df[df.type_of_service == 'Psychotherapy']
##Create new dataset that only includes Skilled Nursing Facility
df_nurse=df[df.type_of_service == 'Skilled Nursing Facility']
##Create new dataset that only includes Telemedicine
df_tele=df[df.type_of_service == 'Telemedicine']

##View Indepentdent Diagnostic Testing Facility Pt A and Preventative Health Correlation
df_testA_corr=df_testA.corr()
df_prevent_corr=df_prevent.corr()
df_nurse_corr=df_nurse.corr()

##Create 3 new comparitive columns for further analysis
df['payments_per_user']=df['total_payment']/df['num_of_users']
df['payments_per_person']=df['total_payment']/df['population']
df['people_per_user']=df['population']/df['num_of_users']

##Create 3 "Big City" datasets
df_boston=df[(df.cbsa_name_x=='Boston-Cambridge-Newton, MA-NH')]
boston_year=df_boston.groupby(['year']).mean()
df_phoenix=df[(df.cbsa_name_x=='Phoenix-Mesa-Scottsdale, AZ')]
phoenix_year=df_phoenix.groupby(['year']).mean()
df_sanFran=df[(df.cbsa_name_x=='San Francisco-Oakland-Hayward, CA')]
sanFran_year=df_sanFran.groupby(['year']).mean()

##Create 3 "Medium to Small City" datasets
df_worcester=df[(df.cbsa_name_x=='Worcester, MA-CT')]
worcester_year=df_worcester.groupby(['year']).mean()
df_omaha=df[(df.cbsa_name_x=='Omaha-Council Bluffs, NE-IA')]
omaha_year=df_omaha.groupby(['year']).mean()
df_greenville=df[(df.cbsa_name_x=='Greenville-Anderson-Mauldin, SC')]
greenville_year=df_greenville.groupby(['year']).mean()

##Begin Plotting

##Create a heat map (Descriptive)
sns.heatmap(df_corr,annot=True)

##Create bar graph showing which services have the most Providers
plt.figure(figsize=(9,6))
plt.barh(df_services_providers['type_of_service'],df_services_providers['num_of_providers'])
plt.title('Providers per Service')
plt.xlabel('Total Number of Providers', horizontalalignment='center')
plt.xticks(rotation=90)
plt.show()

##Create bar graph showing which services have the most users
plt.figure(figsize=(9,6))
plt.barh(df_services_users['type_of_service'],df_services_users['num_of_users'])
plt.title('Users per Service')
plt.xlabel('Total Number of Users', horizontalalignment='center')
plt.xticks(rotation=90)
plt.show()

##Create a bar graph showing which services produce the most revenue
plt.figure(figsize=(9,6))
plt.barh(df_services_payments['type_of_service'],df_services_payments['total_payment'])
plt.title('Payments per User')
plt.xlabel('Total Payment Amount', horizontalalignment='center')
plt.xticks(rotation=90)
plt.show()

##Create a scatter plot showing a relationship with Independent Diagnostic Testing Facility Pt A
plt.figure(figsize=(9,6))
plt.scatter(df['num_of_users'],df['total_payment'],marker='.',color='blue')
plt.title("")
plt.xlabel("Users")
plt.ylabel("Payment")
m,b=np.polyfit(df_amb1['num_of_users'], df_amb1['total_payment'],1)
plt.plot(df_amb1['num_of_users'], m*df_amb1['num_of_users']+b,color='gray')
m,b=np.polyfit(df_amb2['num_of_users'], df_amb2['total_payment'],1)
plt.plot(df_amb2['num_of_users'], m*df_amb2['num_of_users']+b,color='gray')
m,b=np.polyfit(df_amb3['num_of_users'], df_amb3['total_payment'],1)
plt.plot(df_amb3['num_of_users'], m*df_amb3['num_of_users']+b,color='gray')
m,b=np.polyfit(df_card['num_of_users'], df_card['total_payment'],1)
plt.plot(df_card['num_of_users'], m*df_card['num_of_users']+b,color='gray')
m,b=np.polyfit(df_chiro['num_of_users'], df_chiro['total_payment'],1)
plt.plot(df_chiro['num_of_users'], m*df_chiro['num_of_users']+b,color='gray')
m,b=np.polyfit(df_lab['num_of_users'], df_lab['total_payment'],1)
plt.plot(df_lab['num_of_users'], m*df_lab['num_of_users']+b,color='orange', label='Clinical Laboratory')
m,b=np.polyfit(df_dialysis['num_of_users'], df_dialysis['total_payment'],1)
plt.plot(df_dialysis['num_of_users'], m*df_dialysis['num_of_users']+b,color='gray')
m,b=np.polyfit(df_fqhc['num_of_users'], df_fqhc['total_payment'],1)
plt.plot(df_fqhc['num_of_users'], m*df_fqhc['num_of_users']+b,color='gray')
m,b=np.polyfit(df_home['num_of_users'], df_home['total_payment'],1)
plt.plot(df_home['num_of_users'], m*df_home['num_of_users']+b,color='gray')
m,b=np.polyfit(df_hospice['num_of_users'], df_hospice['total_payment'],1)
plt.plot(df_hospice['num_of_users'], m*df_hospice['num_of_users']+b,color='gray')
m,b=np.polyfit(df_testA['num_of_users'], df_testA['total_payment'],1)
plt.plot(df_testA['num_of_users'], m*df_testA['num_of_users']+b,color='green', label='Diagnostic Testing A')
m,b=np.polyfit(df_testB['num_of_users'], df_testB['total_payment'],1)
plt.plot(df_testB['num_of_users'], m*df_testB['num_of_users']+b,color='gray')
m,b=np.polyfit(df_ophtho['num_of_users'], df_ophtho['total_payment'],1)
plt.plot(df_ophtho['num_of_users'], m*df_ophtho['num_of_users']+b,color='red', label='Ophthamology')
m,b=np.polyfit(df_ltc['num_of_users'], df_ltc['total_payment'],1)
plt.plot(df_ltc['num_of_users'], m*df_ltc['num_of_users']+b,color='gray')
m,b=np.polyfit(df_ptot['num_of_users'], df_ptot['total_payment'],1)
plt.plot(df_ptot['num_of_users'], m*df_ptot['num_of_users']+b,color='gray')
m,b=np.polyfit(df_prevent['num_of_users'], df_prevent['total_payment'],1)
plt.plot(df_prevent['num_of_users'], m*df_prevent['num_of_users']+b,color='yellow', label='Preventative Medicine')
m,b=np.polyfit(df_psych['num_of_users'], df_psych['total_payment'],1)
plt.plot(df_psych['num_of_users'], m*df_psych['num_of_users']+b,color='gray')
m,b=np.polyfit(df_nurse['num_of_users'], df_nurse['total_payment'],1)
plt.plot(df_nurse['num_of_users'], m*df_nurse['num_of_users']+b,color='purple', label='Skilled Nursing Facility')
m,b=np.polyfit(df_tele['num_of_users'], df_tele['total_payment'],1)
plt.plot(df_tele['num_of_users'], m*df_tele['num_of_users']+b,color='gray')
plt.title('Correlations Between Userbase and Payments per Service')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

##Create a heat map
sns.heatmap(df_testA_corr,annot=True)
plt.title('Diagnostic Testing Part A')

##Create a heat map
sns.heatmap(df_prevent_corr,annot=True)
plt.title('Preventative Health')

##Create a heat map
sns.heatmap(df_nurse_corr,annot=True)
plt.title('Skilled Nursing')

##Create a bar graph showing which services produce the most revenue per city
plt.figure(figsize=(9,6))
plt.barh(df_boston['type_of_service'],df_boston['total_payment'], color='blue', label='Boston')
plt.barh(df_sanFran['type_of_service'],df_sanFran['total_payment'], color='yellow', label='San Francisco')
plt.barh(df_phoenix['type_of_service'],df_phoenix['total_payment'], color='red', label='Phoenix')
plt.title('Payments per Service')
plt.xlabel('Total Payment Amount')
plt.legend()
plt.xticks(rotation=90)
plt.show()

##Create a bar graph showing which services have the most users per city
plt.figure(figsize=(9,6))
plt.barh(df_boston['type_of_service'],df_boston['num_of_users'], color='blue', label='Boston')
plt.barh(df_sanFran['type_of_service'],df_sanFran['num_of_users'], color='yellow', label='San Francisco')
plt.barh(df_phoenix['type_of_service'],df_phoenix['num_of_users'], color= 'red', label='Phoenix')
plt.title('Users per Service')
plt.xlabel('Total Users', horizontalalignment='center')
plt.legend()
plt.xticks(rotation=90)
plt.show()

##Create a bar graph showing which services produce the most revenue per city
plt.figure(figsize=(9,6))
plt.barh(df_worcester['type_of_service'],df_worcester['total_payment'] ,color='orange', label='Worcester')
plt.barh(df_greenville['type_of_service'],df_greenville['total_payment'], color='green', label='Greenville')
plt.barh(df_omaha['type_of_service'],df_omaha['total_payment'], color='black', label='Omaha')
plt.title('Payments per Service')
plt.xlabel('Total Payment Amount', horizontalalignment='center')
plt.legend()
plt.xticks(rotation=90)
plt.show()

##Create a bar graph showing which services have the most users per city
plt.figure(figsize=(9,6))
plt.barh(df_worcester['type_of_service'],df_worcester['num_of_users'] ,color='orange', label='Worcester')
plt.barh(df_greenville['type_of_service'],df_greenville['num_of_users'], color='green', label='Greenville')
plt.barh(df_omaha['type_of_service'],df_omaha['num_of_users'], color='black', label='Omaha')
plt.title('Users per Service')
plt.xlabel('Total Users', horizontalalignment='center')
plt.legend()
plt.xticks(rotation=90)
plt.show()

##Create a bar graph showing which services produce the most revenue per city
plt.figure(figsize=(9,6))
plt.barh(df_boston['type_of_service'],df_boston['total_payment'], color='blue', label='Boston')
plt.barh(df_worcester['type_of_service'],df_worcester['total_payment'], color='orange', label='Worcester')
plt.title('Payments per Service')
plt.xlabel('Total Payment Amount')
plt.legend()
plt.xticks(rotation=90)
plt.show()

##Create a bar graph showing which services have the most users per city
plt.figure(figsize=(9,6))
plt.barh(df_boston['type_of_service'] ,df_boston['num_of_users'], color='blue', label='Boston')
plt.barh(df_worcester['type_of_service'],df_worcester['num_of_users'], color='orange', label='Worcester')
plt.title('Users per Service')
plt.xlabel('Total Users', horizontalalignment='center')
plt.legend()
plt.xticks(rotation=90)
plt.show()