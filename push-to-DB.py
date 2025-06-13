from sqlalchemy import create_engine
from urllib.parse import quote_plus
from sqlalchemy.exc import OperationalError
import pandas as pd

# Initialize lists
d = []
t = []
gap = []
grp = []
v = []
gi = []
sm1 = []
sm2 = []
sm3 = []

# Read file
with open(r'C:\Users\rav80478\Desktop\project-py\Household-Energy-forecasting\household_power_consumption.txt') as txt_file:
    for index, data in enumerate(txt_file):
        if index == 0:
            continue  # skip header
        items = data.strip().split(";")
        if len(items) == 9:
            d.append(items[0])
            t.append(items[1])
            gap.append(items[2])
            grp.append(items[3])
            v.append(items[4])
            gi.append(items[5])
            sm1.append(items[6])
            sm2.append(items[7])
            sm3.append(items[8])

# Create DataFrame
df = pd.DataFrame({
    "Date": d,
    "Time": t,
    "Global_active_power": gap,
    "Global_reactive_power": grp,
    "Voltage": v,
    "Global_intensity": gi,
    "sub_metering_1": sm1,
    "sub_metering_2": sm2,
    "sub_metering_3": sm3
})

# Preprocessing
#df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
#df.replace('?', pd.NA, inplace=True)
#for col in ['Global_active_power', 'Global_reactive_power', 'Voltage',
#            'Global_intensity', 'sub_metering_1', 'sub_metering_2', 'sub_metering_3']:
#    df[col] = pd.to_numeric(df[col], errors='coerce')
#df.dropna(inplace=True)
#df.drop(columns=['Date', 'Time'], inplace=True)
df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')


# DB connection
password = quote_plus("Chennai@1234")
engine = create_engine(f"mysql+pymysql://root:{password}@localhost/power_house")

# Push to MySQL
try:
    df.to_sql('dataset', engine, index=False, if_exists='replace')
    print("Successfully inserted into MySQL.")
except OperationalError as e:
    print("OperationalError occurred:", e)


# In[ ]: