import pandas as pd
from sqlmodel import Field, SQLModel, create_engine, Session
from typing import Optional
from keys import DATABASE_URL

df = pd.read_csv("Dataset/database_indo.csv")

class Car(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    maker: str
    model: str
    number_of_cylinders: int
    engine_type: int
    engine_horse_power: float
    engine_horse_power_rpm: int
    transmission: int
    fuel_tank_capacity: int
    acceleration_0_to_100_km: float
    max_speed_km_per_hour: int
    fuel_grade: int
    year: int
    type_of_car: int
    car_name: str

engine = create_engine(DATABASE_URL)
SQLModel.metadata.create_all(engine)

with Session(engine) as session:
    for index, row in df.iterrows():
        try:
            car = Car(
                maker=row["Maker"],
                model=row["Model"],
                number_of_cylinders=row["Number_of_Cylinders"],
                engine_type=row["Engine_Type"],
                engine_horse_power=row["Engine_Horse_Power"],
                engine_horse_power_rpm=row["Engine_Horse_Power_RPM"],
                transmission=row["Transmission"],
                fuel_tank_capacity=row["Fuel_Tank_Capacity"],
                acceleration_0_to_100_km=row["Acceleration_0_to_100_Km"],
                max_speed_km_per_hour=row["Max_Speed_Km_per_Hour"],
                fuel_grade=row["Fuel_Grade"],
                year=row["Year"],
                type_of_car=row["Type_of_Car"],
                car_name=row["Car Name"],
            )
            session.add(car)
            session.commit()
        except:
            pass