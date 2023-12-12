import pandas as pd

#Defining used columns for training
used_columns = ['Make',
                'Modle',
                'Year_from',
                'Year_to',
                'Body_type',
                'number_of_cylinders',
                'engine_type',
                'engine_hp',
                'engine_hp_rpm',
                'transmission',
                'mixed_fuel_consumption_per_100_km_l',
                'fuel_tank_capacity_l',
                'acceleration_0_100_km/h_s',
                'max_speed_km_per_h',
                'fuel_grade',
                'battery_capacity_KW_per_h',
                'charging_time_h',
                'electric_range_km']

#Reading .csv data
df = pd.read_csv('Dataset/Raw Data/Car Dataset 1945-2020.csv',
                 usecols=used_columns,
                 low_memory=False)
df.rename(columns={'Modle':'Model'}, inplace=True)

#Using average year on the data
df['year'] = df[['Year_from', 'Year_to']].mean(axis=1).round()

#Cleaning some Null datas and more unused columns
df = df[df['battery_capacity_KW_per_h'].isna()]
df = df[df['charging_time_h'].isna()]
df = df[df['electric_range_km'].isna()]
df = df[df['Year_from'].notna()]
df = df[df['Year_to'].notna()]
df.drop(columns=['Year_from',
                 'Year_to',
                 'battery_capacity_KW_per_h',
                 'charging_time_h',
                 'electric_range_km'],
                 inplace=True)

#Creating new columns of car type from the body type data
df['car_type'] = df['Body_type'].replace({
    'Cabriolet': 'Sedan',
    'Coupe': 'Sedan',
    'Roadster': 'Sedan',
    'Sedan': 'Sedan',
    'Crossover': 'Jeep',
    'Hatchback': 'Sedan',
    'Liftback': 'Sedan',
    'Wagon': 'Sedan',
    'Minivan': 'Bus',
    'Fastback': 'Sedan',
    'Pickup': 'PickUp',
    'Targa': 'Sedan',
    'hardtop': 'Sedan',
    'Limousine': 'Bus',
    'Hatchback 3 doors': 'Sedan'
})
df.drop(columns=['Body_type'], inplace=True)
df.dropna(inplace=True)

#Using only cars from 2009 and above
df = df[df['year'] >= 2000]

#Encoding the car type data
df['car_type'] = df['car_type'].replace({
    'Sedan': 0,
    'Jeep': 1,
    'Bus': 2,
    'PickUp': 3
})
df['car_type'] = df['car_type'].astype(int)

#Processing the engine type data
not_used_engine = ['Hybrid',
                   'Gasoline, Electric',
                   'Diesel, Hybrid']
df = df[~df['engine_type'].isin(not_used_engine)]
df['engine_type'] = df['engine_type'].replace({
    'Gasoline, Gas': 'Gasoline',
    'Gas': 'Gasoline',
})
df.loc[:,'engine_type'] = df['engine_type'].replace({
    'Gasoline': 0,
    'Diesel': 1,
})
df['engine_type'] = df['engine_type'].astype(int)

#Processing fuel grade data
not_used_fuel = ['Ethanol', '80']
df = df[~df['fuel_grade'].isin(not_used_fuel)]
df['fuel_grade'] = df['fuel_grade'].replace({
    '95': '3',
    '92': '2',
    '98': '3',
    'diesel': '0',
    '95, 92': '2',
    '98, 95': '3',
    'Gasoline': '1',
    'Gas': '1',
    '95, Ethanol': '3',
    '98, 95, 92': '3',
    '95, Gas': '3',
    '92, Ethanol': '3',
    'Gasoline, Gas': '1',
    '92, Gas': '2',
    '98, Gas': '3',
    '95, 92, Gas': '2',
    '98, 95, Gas': '3',
    '98, 95, 92, Ethanol': '3',
    '98, Ethanol': '3'
})
df['fuel_grade'] = df['fuel_grade'].astype(int)

#Processing maximum speed data
df['max_speed_km_per_h'] = df['max_speed_km_per_h'].str.replace(',', '.').astype(float).astype(int)
df['max_speed_km_per_h'] = df['max_speed_km_per_h'].astype(int).round()

#Processing transmission type data
not_used_transmission = ['robot',
                         'Continuously variable transmission (CVT)']
df = df[~df['transmission'].isin(not_used_transmission)]
df['transmission'] = df['transmission'].replace({
    'Manual': '1',
    'Automatic': '0'
})
df['transmission'] = df['transmission'].astype(int)

#Changing data types into integer for some columns
df['fuel_tank_capacity_l'] = df['fuel_tank_capacity_l'].str.replace(',', '.').astype(float).astype(int)
df['engine_hp_rpm'] = df['engine_hp_rpm'].astype(int)
df['number_of_cylinders'] = df['number_of_cylinders'].astype(int)

#Renaming all columns to match the indonesian car train set
df.rename(columns={'Make':'Maker',
                   'number_of_cylinders':'Number_of_Cylinders',
                   'engine_type':'Engine_Type',
                   'engine_hp':'Engine_Horse_Power',
                   'engine_hp_rpm':'Engine_Horse_Power_RPM',
                   'transmission':'Transmission',
                   'mixed_fuel_consumption_per_100_km_l':'Mixed_Fuel_Consumption_per_100_km_l',
                   'fuel_tank_capacity_l':'Fuel_Tank_Capacity',
                   'acceleration_0_100_km/h_s':'Acceleration_0_to_100_Km',
                   'max_speed_km_per_h':'Max_Speed_Km_per_Hour',
                   'fuel_grade':'Fuel_Grade',
                   'year':'Year',
                   'car_type':'Type_of_Car'}, inplace=True)

print(df.info())
print('')
print(df.head())

#Saving training dataset
df.to_csv('Dataset/train_data_cleaned.csv', index=False)