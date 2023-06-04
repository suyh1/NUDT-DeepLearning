# coding=utf-8
from __future__ import absolute_import, division, print_function, unicode_literals

from flight_cls_todo import test

result_dict = test()

if result_dict['c_pred_accuracy'] > 0.95 and result_dict['lm_pred_mean_squared_error'] < 3.0:
    print("Success!")
else:
    print("Something wrong!")