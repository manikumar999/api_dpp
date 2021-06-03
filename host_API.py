# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 11:19:42 2021

@author: Hp
"""

"""This module performs the optimzation algorithm.
        
    Converts the json strings into dataframes,
    performs optimization algorithm on dataframes,
    converts datframes into json strings.
        
    Parameters
    ----------
    param1: json string that includes the dataframes.
        
    Returns
    -------
    Three json strings including information about dataframes.
        
    """
import json
import pandas as pd

#import dataiku
import numpy as np
#from dataiku import pandasutils as pdu

import pulp as pp

def api_py_function(param1):
    """Performs the entire optimzation algorithm.
        
    Converts the json strings into dataframes,
    performs optimization algorithm on dataframes,
    converts datframes into json strings.
        
    Parameters
    ----------
    param1: json string that includes the dataframes.
        
    Returns
    -------
    Two dataframes required for the optimization algorithm.
        
    """    
    #dataiku.set_remote_dss("https://d02ydkudes001.sabic.com/", "iFpftesAw2NdFYm0Iz7IoTdGzEB1R9AC", 
                           #no_check_certificate=True)
                
    def create_df(param1):
        """Converts json strings into dataframes.
        
        Converts the json strings into appropriate dataframes
        that are used an input for the optimization algorithm
        
        Parameters
        ----------
        param1: json string that includes the dataframes.
        
        Returns
        -------
        Three json strings including information about dataframes.
        
        """
        param1 = str(param1)
        json_acceptable_string = param1.replace("'", "\"")
        json_obj=json.loads(json_acceptable_string)
        
        if "Plant_data" not in json_obj:
            print("Please Check the naming format convention for the json string input for Plant data")
            data_plant = pd.DataFrame()
            
        else:
            data_plant = pd.DataFrame(json_obj["Plant_data"])
            data_plant.rename(columns={"WarehouseCapacity":"warehouse_capacity",
                                       "SafetyStock":"safety_stock",
                                       "OpeningStock":"opening_stock",
                                       "Period":"Period",
                                       "on_route":"on_route",
                                       "predicted_consumption_mp_level":"predicted_consumption_mp_level"},
                              inplace=True)
            
        try:
            supplier_df = (pd.DataFrame(json_obj["Suppliers"]["Spplier_Name"]["Name"]))
            column_order = ["Month1","Month2","Month3","Month4","Month5",
                            "Month6","SupplierId","SupplierName","TimePeriod"]
            supplier_df = supplier_df[column_order]
            supplier_df_pivot = supplier_df.pivot_table(columns=["SupplierName"],
                                                        index=["TimePeriod"],
                                                        values=supplier_df.columns[0:-3],
                                                        aggfunc="min").T
            
        except Exception as x:
            print("Please Check the naming format convention for the json string input for Supplier data")
            supplier_df_pivot = pd.DataFrame()
            return data_plant,supplier_df_pivot
        
        supplier_df_pivot.reset_index(inplace=True)
        supplier_df_pivot.rename(columns={"level_0":supplier_df_pivot.columns.name},inplace=True)
        supplier_df_pivot.columns.name=None
        supplier_df_pivot.rename(columns={"TimePeriod":"Period",
                                          "SupplierName":"Supplier",
                                          "Supplier Max. Capacity":"MaxSupply",
                                          "Forecasted Price":"Price",
                                          "Lead Time":"Lead_time"},
                                 inplace=True)
        supplier_df_pivot[["MaxSupply","Price"]] = supplier_df_pivot[["MaxSupply","Price"]].astype("float")
        supplier_df_pivot[["Lead_time"]] = supplier_df_pivot[["Lead_time"]].astype("int")
        
        return data_plant,supplier_df_pivot
        
    data_plant_opt,new_data_supplier_opt = create_df(param1)
    
    def linear_opt(data_plant_opt, new_data_supplier_opt):
        """Performs optimization on dataframes.
        
        The values required for algorithm are calculated,
        constraints are defined, recommendations are defined,
        optimal values are derived.
        
        Parameters
        ----------
        data_plant_opt: Dataframe including plant details.
        new_data_supplier_opt: Dataframe including supplier details.
        
        Returns
        -------
        Three dataframes generated after the optimization algorithm.
        
        """
        if data_plant_opt.empty or new_data_supplier_opt.empty:
            error_msg = "Could not proceed with optimization"
            print(error_msg)
            Solution = pd.DataFrame({"Error_info":[error_msg]})
            SolutionPrice =  pd.DataFrame({"Error_info":[error_msg]})
            Error_df =  pd.DataFrame({"Error_info":[error_msg]})
            return Solution,SolutionPrice,Error_df
        
        try:
            data_plant = data_plant_opt.copy()
        except Exception as x:
            print("making copy of plant data is unsuccessfull")

        try:
            current_period = data_plant.iloc[:1,:]    
            PlantOpenStock = current_period.opening_stock + current_period.on_route 
            PlantOpenStock = PlantOpenStock - current_period.predicted_consumption_mp_level
        except Exception as x:
            print("Something went wrong when computing plant opening stock")
            
        try:
            data_plant = data_plant.iloc[1:,:].reset_index(drop=True)
        except Exception as x:
            print("Subsetting plant dataframe is unsuccessfull")

        try:
            new_data_supplier = new_data_supplier_opt.copy()
        except Exception as x:
            print("making copy of new data supplier is unsuccessfull")
        
        try:
            new_data_supplier = new_data_supplier.groupby(
                ['Period'],sort=True,group_keys=True).apply(
                lambda x: x.sort_values(['Supplier'], ascending=False)).reset_index(drop=True)
            new_data_supplier.reset_index(drop=True,inplace=True)
        except Exception as x:
            print("Something went wrong when sorting new data supplier")
        
        nSup = len(new_data_supplier['Supplier'].unique())
        new_data_supplier = new_data_supplier.iloc[nSup:].reset_index(drop=True)
        sup_list = list(new_data_supplier['Supplier'].unique())
        model = pp.LpProblem("Order_Planning", pp.LpMinimize)
        nMnth=data_plant.index

        try:
            deliver_qty = pp.LpVariable.dicts("To_deliver", [t for t in range(0,nSup*len(nMnth))],
                                              lowBound=0, cat="Continuous")
            opening_stock_qty = pp.LpVariable.dicts("Opening_stock",[t for t in nMnth],
                                                    lowBound=0, cat="Continuous")
            closing_stock_qty = pp.LpVariable.dicts("Closing_Stock",[t for t in nMnth],
                                                    lowBound=0, cat="Continuous")
            total_arrival_qty =  pp.LpVariable.dicts("Total_arrival",[t for t in nMnth],
                                                     lowBound=0, cat="Continuous")
        except Exception as x:
            print("Declaration of decision varaibles is unsuccessfull")

        delivered_obj = [i for i in deliver_qty.values()]
        price_obj = new_data_supplier.Price.values.tolist()
        
        try:
            obj_func = pp.lpSum([price_obj[i]*delivered_obj[i] for i in range(0,len(delivered_obj))])
        except Exception as x:
            print("Objective function declaration failed")

        model +=  obj_func 
        try:
            MaxSupply = list(new_data_supplier.MaxSupply.values)
            for i in range(0,len(delivered_obj)):
                model +=  delivered_obj[i] - MaxSupply[i] <=0
        except Exception as x:
            print("Failed while defining delivery constraints w.r.t max supply")
        
        try:
            index_list = [index for index in range(0,len(deliver_qty),nSup)]
            index_list.append(index_list[-1]+nSup)
            for i in range(0,len(total_arrival_qty)):
                l = []
                for j in range(index_list[i],index_list[i+1]):
                    l.append(deliver_qty[j])
                model +=  total_arrival_qty[i] == pp.lpSum(l)
        except Exception as x:
            print("Something went wrong while defining total arrival \
            equality constraints w.r.t delivery quantities")

        leadtime_dict = dict(zip(new_data_supplier[:nSup]['Supplier'].values.tolist(),
                                 new_data_supplier[:nSup]['Lead_time'].values.tolist()))
        leadtime = list(leadtime_dict.values())
        
        try:
            lead_index = []
            for i in leadtime_dict.keys():
                leadtime = leadtime_dict[i]
                if leadtime!=0:
                    cnt=0
                    for j in (new_data_supplier[new_data_supplier.Supplier==i]).index:
                        if cnt==leadtime:
                            break
                        lead_index.append(j)
                        model += deliver_qty[j] == 0
                        cnt+=1
        except Exception as x:
            print("Something went wrong while defining delivery constraints w.r.t lead time values")

        model += opening_stock_qty[0] == PlantOpenStock
        for i in range(0,data_plant.shape[0]):
            model +=  closing_stock_qty[i] - data_plant['safety_stock'][i] >= 0

        for i in range(0,data_plant.shape[0]):
            model +=  opening_stock_qty[i] - data_plant['safety_stock'][i] >= 0

        for i in range(0,data_plant.shape[0]):
            model +=  opening_stock_qty[
                i]+ data_plant['on_route'][i] + total_arrival_qty[
                i] - data_plant['warehouse_capacity'][i]<=0

        for i in range(0,len(opening_stock_qty)-1):
            model +=  opening_stock_qty[i+1] - closing_stock_qty[i] == 0

        for i in range(0,data_plant.shape[0]):
            model += opening_stock_qty[i] + total_arrival_qty[
                i] - data_plant['predicted_consumption_mp_level'][
                i] + data_plant['on_route'][i] - closing_stock_qty[i] == 0

        result=model.solve()        
        print("Status: " + pp.LpStatus[result])
        
        A = np.array([])
        for v in model.variables():            
            A = np.append(A,v.varValue)
            
        try:
            s = pd.Series(A[-len(total_arrival_qty):])                
            final_data_plant= data_plant.copy()        
            final_data_plant['closing_stock'] = A[:data_plant.shape[0]]
            final_data_plant['opening_stock'] = A[data_plant.shape[0]:2*data_plant.shape[0]]
        except Exception as x:
            print("Something went wrong while rearranging the final_data_plant \
            values corresponding to the array A")
        Solution = pd.concat([final_data_plant,pd.DataFrame({"to_arrive":s})],axis=1)
        
        try:
            deliver_dict = {}
            for v in model.variables():
                if v.name.startswith("To_deliver"):
                    deliver_dict[v.name] = v.varValue
        except Exception as x:
            print("Extraction of To_deliver values w.r.t deliver quantities has failed")
        
        try:
            delivery_keys = sorted(deliver_dict)
            
            delivery_keys.sort(key = lambda x: int(x.split("_")[-1]))
        except Exception as x:
            print("Error occured while sorting delivery_keys")
            
        sorted_deliver_values = np.array([])
        for i in delivery_keys:
            sorted_deliver_values = np.append(sorted_deliver_values,deliver_dict[i])
        
        try:
            SolutionPrice = pd.DataFrame(columns=["Period"]+sup_list,index=range(0,data_plant.shape[0]))
            SolutionPrice["Period"] = data_plant.Period
            SolutionPrice.loc[:,1:] = sorted_deliver_values.reshape(data_plant.shape[0],nSup,order='C')
    
            SolutionPrice["Total_Monthly_Quantity"] = SolutionPrice.iloc[:,1:].sum(axis=1)
            SolutionPrice.loc["SupplierTotal"] =  SolutionPrice.iloc[:,1:].sum(axis=0)
            SolutionPrice.reset_index(drop=True,inplace=True)
            SolutionPrice.iloc[-1,0] = "SupplierTotal"
        except Exception as x:
            print("Error occured while creating dataframe SolutionPrice")

        #---------------------Error Matrix---------------------------------------------------------

        Error_list = []
        
        if pp.LpStatus[result] == "Optimal":
            Error_list.append("Optimal solution")
            Error_df = pd.DataFrame(pd.Series(Error_list),columns=["Error_info"])

        else:
            try:
                Error_list.append("Infeasible solution")
                min_leadtime = min(leadtime_dict.values())
                wc_index = []
                warehouse_violation = []
                for i in range(0, data_plant.shape[0]):
                    warehouse_check = Solution.opening_stock[
                        i] + Solution.on_route[i] <= Solution.warehouse_capacity[i]
                    if not warehouse_check:
                        wc_index.append(i)
                        warehouse_violation.append(round(
                            Solution.opening_stock[i] + Solution.on_route[
                                i] - Solution.warehouse_capacity[i],2))
                if len(wc_index)!=0:
                    Error_list.append("index {} warehouse capacity violated, reduce quantity by {}"\
                                      .format(wc_index,warehouse_violation))
            except Exception as x:
                print("Something went wrong while checking warehouse \
                capacity violation for infeasibility")


            # Initial opening stock check
            
            try:
                opening_stock_const = data_plant['safety_stock'][
                    0] - data_plant['on_route'][
                    0] + data_plant['predicted_consumption_mp_level'][0]
                opening_stock_value = PlantOpenStock
                os_diff = opening_stock_const - opening_stock_value
                if os_diff[0] > 0:
                    Error_list.append("opening stock is insufficient, needs at least {:} more to meet \
                    the first condition".format(round(os_diff[0],2)))
                    
            except Exception as x:
                print("Some error occured while checking safety stock violation w.r.t initial opening stock")

            mnth_dict = dict(zip(data_plant.index,data_plant.Period))
            leadtime_dict = dict(zip(new_data_supplier['Supplier'].values.tolist(),
                                     new_data_supplier['Lead_time'].values.tolist()))
    
            # Initial safety check
            
            try:
                safety_diff = 0
                for i in range(1,min_leadtime+1):
                    initial_consumption = PlantOpenStock + data_plant.on_route[
                        i-1] - data_plant.predicted_consumption_mp_level[
                        i-1] + data_plant.on_route[
                        i] - Solution.predicted_consumption_mp_level[i]
                    
                    initial_safety_stock = Solution.safety_stock[i]
                    
                    if initial_safety_stock > initial_consumption[0]:
                        safety_diff += round((initial_safety_stock - initial_consumption[0]),2)
                if safety_diff !=0:
                    Error_list.append('Safety stock is not maintained for month with minimum lead time =>\
                    increase the quantity for any of opening_stock or on_route by {} before {}'\
                                      .format(round(safety_diff,2),mnth_dict[min_leadtime+1]))

            except Exception as x:
                print("Error occured while checking safety stock violation till the month corresponding \
                to minimum lead time")
           

            try:                
                if pp.LpStatus[result] == 'Infeasible':
                    for i in lead_index:
                        if sorted_deliver_values[i] != 0:                                                        
                            supply_inc = sorted_deliver_values[i]                            
                            Error_list.append("to_deilver_{} affected, increase the on route quantity \
                            by {} before {} ".format(i,supply_inc,mnth_dict[i//nSup +1]))
                            
            except Exception as x:
                print("Failed to check to_deliver values if affected for infeasible solution")            

            def identify_suppliers(leadtime):
                """Identifies supplier names.        
        
                Parameters
                ----------
                leadtime: for identifying the preceding suppliers corresponding to the leadtime.
        
                Returns
                -------
                list of suppliers.
        
                """
                sups = []
                for j in leadtime_dict.items():
                    if j[1] <= leadtime:
                        sups.append(j[0])
                return sups

            total_safety_diff = 0
            idx_cnt= 0
            index_list = []
            
            try:
                for i in range(min_leadtime,data_plant.shape[0]-1):    
                    safety_check = Solution.closing_stock[i] - Solution.predicted_consumption_mp_level[i+1] +\
                                   Solution.on_route[i+1] >= data_plant['safety_stock'][i+1]
                    if not safety_check:
                        index_list.append(str(i+1))
                        idx_cnt+=1    
                        safety_diff = Solution.safety_stock[i+1]-(Solution.closing_stock[i]-\
                                                                  Solution.predicted_consumption_mp_level[i+1] +\
                                                                  Solution.on_route[i+1])                        
                        if idx_cnt ==1:
                            mnth = mnth_dict[i]
                            sups = identify_suppliers(i)    
                        total_safety_diff += round(safety_diff,2)
                if total_safety_diff !=0:                                        
                    Error_list.append("Please either increase the on route quantity OR\
                    maximum supply of suppliers {} by at least {} for month {}, Safety stock of next month is violated, "\
                                      .format(' or '.join(sups),round(total_safety_diff,2),mnth))
            except Exception as x:
                print("Error caught while testing safety stock violation after the month \
                corresponding to minimum lead time")

            Error_df = pd.DataFrame(pd.Series(Error_list),columns=["Error_info"])

        return Solution,SolutionPrice,Error_df
    
    Solution,SolutionPrice,Error_df = linear_opt(data_plant_opt, new_data_supplier_opt)
    
    def json_output(Solution,SolutionPrice,Error_df):
        """Converts dataframes into json strings.
        
        Converts the dataframes into json strings
        that are output of this optimization algorithm.
        
        Parameters
        ----------
        Solution: Dataframe containing information about solution.
        SolutionPrice: Dataframe containing about pricing.
        Error_df: Dataframe containing information about recommendations.
        
        Returns
        -------
        Three json strings including information about dataframes.
        
        """
        Solution_json = Solution.to_json()
        Solution_price_json = SolutionPrice.to_json()
        Error_df = Error_df.to_json()
        
        return Solution_json,Solution_price_json,Error_df
         
    final_solution, final_solution_price, final_Error_df = json_output(Solution,SolutionPrice,Error_df)  
    return final_solution, final_solution_price,final_Error_df