import os

import gspread
from oauth2client.service_account import ServiceAccountCredentials


def get_google_spreadsheet(service_account_dir, service_account_json, spreadsheet_key, spreadsheet_tab_name):
    scope = ['https://spreadsheets.google.com/feeds']
    # Change to directory with service account credentials json
    os.chdir(service_account_dir)
    # Get service account credentials from json file
    service_account_credentials = ServiceAccountCredentials.from_json_keyfile_name(service_account_json, scope)
    # Get spreadsheet
    client_obj = gspread.authorize(service_account_credentials)
    spreadsheet_obj = client_obj.open_by_key(spreadsheet_key)
    worksheet = spreadsheet_obj.worksheet(spreadsheet_tab_name)
    return worksheet.get_all_values()

