import pandas as pd
import kookpy
from datetime import datetime, timedelta
import os


def collect_and_save_historical_data(location_name, start_date_str, end_date_str):
    # collects historical surf data, calculates a quality score, and saves it to csv.

    # gecodoe location analysis
    coords = kookpy.geocode_location(location_name)
    if not coords:
        print(f"error: could not find coordinates for {location_name}.")
        return

    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

    all_data = []

    current_date = start_date
    while current_date <= end_date:
        current_date_str = current_date.strftime('%Y-%m-%d')
        print(f"fetching data for {current_date_str}...")

        try:
            # use new api classes for polymorphism demo
            marine_api = kookpy.OpenMeteoMarineAPI(coords['latitude'], coords['longitude'], current_date_str, current_date_str)
            wind_api = kookpy.OpenMeteoWindAPI(coords['latitude'], coords['longitude'], current_date_str, current_date_str)

            marine_data = marine_api.fetch_data()
            wind_data = wind_api.fetch_data()

            # check empty and merge
            if not marine_data.empty and not wind_data.empty:
                combined_df = pd.merge(
                    marine_data, wind_data, on='time', how='inner')

                combined_df['wave_quality_score'] = combined_df.apply(
                    kookpy.calculate_heuristic_score, axis=1)
                # ------------------------------------

                all_data.append(combined_df)
            else:
                print(
                    f"could not fetch data for {current_date_str}. skipping.")
        except Exception as e:
            print(f"error fetching data for {current_date_str}: {e}")

        current_date += timedelta(days=1)

    if all_data:
        full_df = pd.concat(all_data, ignore_index=True)
        # drop missing rows
        full_df.dropna(inplace=True)

        if not full_df.empty:
            # save data to a predictable location, relative to the project root
            file_path = os.path.join('ai', 'historical_surf_data.csv')
            full_df.to_csv(file_path, index=False)
            print(
                f"\nsuccessfully collected and saved {len(full_df)} data points to {file_path}")
        else:
            print("\nno data was collected.")
    else:
        print("\nno data was collected.")


if __name__ == '__main__':
    # run data collection for a year
    location = "laguna beach"
    start = "2023-01-01"
    end = "2024-01-01"
    collect_and_save_historical_data(location, start, end)