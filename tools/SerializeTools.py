# -*- coding: UTF-8 -*-
import json
import sys


def config2json(params_dict, file_url):
    with open(file_url, mode="w+", encoding="utf-8") as file:
        json.dump(params_dict, file, indent=4)
        print("***Rectification parameters are written successfully!***")


def json2config(file_url):
    try:
        with open(file_url, mode="r", encoding="utf-8") as file:
            calibration_params_dict = json.load(file)
            return calibration_params_dict
            print("***Calibration parameters are loaded successfully!***")
    except:
        print("Error: 没有找到文件或读取文件失败")
        sys.exit()
