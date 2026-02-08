import pandas as pd

def engineer_features(df):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp')

    # Power stress
    df['Power_Stress'] = abs(df['Battery_Voltage'] * df['Battery_Current'])

    # Thermal stress
    df['Thermal_Stress'] = (
        df['Battery_Temperature'] + df['Motor_Temperature']
    ) / 2

    # Usage intensity
    df['Usage_Intensity'] = (
        df['Driving_Speed'] * df['Load_Weight']
    )

    # Rolling health degradation
    df['Health_Trend'] = (
        df['Component_Health_Score']
        .rolling(window=10)
        .mean()
        .fillna(method='bfill')
    )

    selected_features = [
        'SoC', 'SoH',
        'Battery_Voltage', 'Battery_Current',
        'Battery_Temperature',
        'Motor_Temperature',
        'Motor_Vibration',
        'Motor_RPM',
        'Brake_Pad_Wear',
        'Power_Stress',
        'Usage_Intensity',
        'Health_Trend'
    ]

    return df[selected_features], df['Failure_Probability'], df['RUL']
