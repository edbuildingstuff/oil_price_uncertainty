Cross, J., Nguyen, B. H., & Tran, T. D. (2022). 
The role of precautionary and speculative demand in the global market for crude oil.
Journal of Applied Econometrics, forthcoming.

-----------------------------------------------------------------------------------------------------------------------------------------------
DATA 
-----------------------------------------------------------------------------------------------------------------------------------------------
The data used in the SVAR model are monthly and titled "Data_SVAR.txt". 
The datasets are arranged in column format with rows corresponding to months and columns corresponding to variables. 
The rows contain data at the following dates ... giving ... total observations per series.
The columns contain data on the following variables (in this order):
C1-the growth rate of global oil production 
C2-the measure of global business cycle
C3-the log of real price of oil
C4-the change in global crude oil inventories 
C5-oil price uncertainty index

The data used in the contruction of the OPU index are titled "Data_SVAR.txt". 
The datasets are arranged in column format with rows corresponding to months and columns corresponding to variables.
The rows contain dates from 1973-January to 2018-June giving 546 total observations per series.
The columns contain data on the following variables (in this order):
C1-Crude oil price: U.S. Crude Oil Imported Acquisition Cost by Refiners 
C2-Energy Index
C3-Crude oil price: Average
C4-Crude oil price: Dubai
C5-Coal price: Australian
C6-Natural gas price: US
C7-Natural gas price: Europe
C8-Natural gas price: Index
C9-Exchange Rate: AUD (DEXUSAL)
C10-Exchange Rate: CAD(EXCAUS)
C11-Exchange Rate: CLP(CCUSSP02CLM650N)
C12-Exchange Rate: NZD( DEXUSNZ)
C13-Exchange Rate: NOK(DEXNOUS)
C14-Exchange Rate: ZAR(EXSFUS)
C15-Economic Activity: Kilian Index
C16-Oil Production
C17-US Uncertainty: JLN
C18-Economic Activity: OECD+6NME Industrial Production
C19-Crude Oil Stock (Thousand Barrels)		
C20-Money Supply: USM1
C21-Price Index: USCPI

Data sources:

All data for the oil market variables in the SVAR was made available from replication files for 
Xiaoqing Zhou, "Refining the Workhorse Oil Market Model", Journal of
Applied Econometrics, Vol. 35, No. 1, 2020, pp. 130-140. 
Link to data archive: http://qed.econ.queensu.ca/jae/datasets/zhou001/

The data for the construction of the OPU index was:
	1- U.S. Crude Oil Imported Acquisition Cost by Refiners (Dollars per Barrel) Data (Source:  U.S. Energy Information Administration (EIA))
	2- Other Fuel Data: Energy Index, Cruide oil index, Coal (Australian), Natural gas (Source: World Bank Pink Sheet) 	
	3- Commodity Exchange Rate: AUD, CAD, CLP, NZD, ZAR (Source: St Louis FED)
	4- Other Macro Variables: 
		i- Real Economic Activity (Source: Kilian, 2009)
		ii- Oil Production  (Source: EIA)
		iii-Above-ground oil inventories (Source: EIA)
		iv- U.S M1, U.S CPI (Source: St Louis FED) 	

-----------------------------------------------------------------------------------------------------------------------------------------------
REPLICATION FILES
-----------------------------------------------------------------------------------------------------------------------------------------------
The data for the OPU index, along with associated replication files are located in the folder called "OilUncertaintyConstruction". 
To replicate the main results run "Figure1.m", "Figure2.m" "Figure3.m", "Figure4.m"
We then store the resulting index to be used in the SVAR model in the folder called "OPU".

The data for the oil market block used in the SVAR is located in the folder called "zhou data" and is taken directly from replication codes for

Zhou, X. (2020). 
Refining the workhorse oil market model. 
Journal of Applied Econometrics, 35(1), 130-140.

To replicate the main results follow these steps:
	1-Obtain draws of 100 SVAR models by running the script "main_1.m" 
		NOTE: This may take a while. To save time, we ran multiple parallel scripts which still took about 1 month on our PC. 
		To save time you can use our stored models and skip straight to step 2.
	2-Combine the obtained draws by running the script "main_2_combine.m". This results a file of combined draws called "IRFS_main_REA.mat".
	3-Use "IRFS_main_REA.mat" to reproduce the
	 - IRFs by running the script "main_3_figure_IRFs.m" and associated table "main_4_table_IRFs.m" 
	 - FEVDs by running the script "main_5_vdc.m"
	 - HDECs by running the script "main_6_hd.m"

