import pandas as pd

ssd_f = pd.read_csv('result_ssd.csv')
res_f = pd.read_csv('result_resnet.csv')
final_res = pd.DataFrame([ssd_f['predicted'], res_f['predicted'], ssd_f['img_name']]).T
final_res.to_csv('result_wait_to_process.csv', header=False)
