#  Bike-sharing-statistics.py
#  The first assignment of Introduction to Data Mining
#  Created by 刘伟 on 2021/5/9.

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

# 字段说明：
# datetime：时间
# season：季节，1=春，2=夏，3=秋，4=冬
# holiday：节假日，0：否，1：是
# workingday：工作日，0：否，1：是
# weather：天气，1:晴天，2:阴天 ，3:小雨或小雪 ，4:恶劣天气（大雨、冰雹、暴风雨或者大雪）
# temp：实际温度，摄氏度
# atemp：体感温度，摄氏度
# humidity：湿度，相对湿度
# windspeed：风速
# casual：未注册用户租借数量
# registered：注册用户租借数量
# count：总租借数量

train = pd.read_csv("./train.csv")
# 提取相关列 & 对时间进行格式化处理
p1 = train[['datetime', 'season', 'holiday', 'workingday', 'count']]
periodDf = p1.copy()
# 日期处理，把日期提取出来（匿名函数分离出来）
periodDf['date'] = periodDf['datetime'].apply(lambda x: x.split()[0])
periodDf['time'] = periodDf['datetime'].apply(lambda x: x.split()[1])
periodDf['year'] = periodDf['date'].apply(lambda x: x.split('-')[0])
periodDf['month'] = periodDf['date'].apply(lambda x: x.split('-')[1])
periodDf['day'] = periodDf['date'].apply(lambda x: x.split('-')[2])
periodDf['hour'] = periodDf['time'].apply(lambda x: x.split(':')[0])
# 星期
periodDf['weekday'] = periodDf['datetime'].apply(lambda x: pd.to_datetime(x).weekday())
# print(periodDf.head())

# 对月份进行划分（区域图）
fig1 = plt.figure(figsize=(16, 8))
ax1 = plt.subplot(1, 1, 1)
# 对数据进行月份和年份的两个特征进行划分
# 通过unstack将month作为行索引，year作为列索引
df1 = periodDf.groupby(['month', 'year']).sum().unstack()['count']
# print(df1)
ax1.set_title('2011-2012 bikes sharing demand by month')
ax1.set_xticks(list(range(12)))
ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'DeC'])
ax1.set_xlim(0, 11)
# "area":折线图与x轴围成的面积区域图
df1.plot(kind='area', ax=ax1)
plt.savefig(r'./区域图.png', dpi=400)

# 对每个季节的小时进行划分（折线图）
df2 = periodDf.groupby(['hour', 'season']).mean().unstack()['count']
df2.columns = ['Spring', 'Summer', 'Autumn', 'Winter']
fig2 = plt.figure(figsize=(16, 8))
ax2 = plt.subplot(1, 1, 1)
ax2.set_title('Bikes sharing demand of four seasons by hour')
ax2.set_xticks(list(range(24)))
ax2.set_xticklabels(list(range(24)))
ax2.set_xlim(0, 23)
df2.plot(kind='line', ax=ax2, style="--.")
plt.savefig(r'./折线图.png', dpi=400)

# 周一到周日的租车数量箱线图
df3 = periodDf[['count', 'weekday']]
fig3 = plt.figure(figsize=(16, 6))
ax3 = plt.subplot(1, 1, 1)
df3.boxplot(by='weekday', ax=ax3, patch_artist=True, notch=False, whis=1.5)
ax3.set_xticklabels(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
ax3.set_ylim(0, 1000)
plt.savefig(r'./箱线图.png', dpi=400)

# 热点图
corr = train.corr()
fig4 = plt.figure(figsize=(10, 10))
ax4 = plt.subplot(1, 1, 1)
ax4.set_title('Heatmap of bikes sharing demand')
sn.heatmap(corr, square=True, annot=True, linewidth=.5, center=2, ax=ax4)
plt.savefig(r'./热点图.png', dpi=400)

# 查看湿度、温度对租车数量的影响，以散点图的方式呈现
# 提取温度和湿度两列的数据
climateDf = train[['temp', 'humidity', 'count']]
fig5 = plt.subplots(1, 2, figsize=(20, 8))

ax5 = plt.subplot(1, 2, 1)
df5 = climateDf[['humidity', 'count']]
# 设置点的位置, s:点的大小, c:点的颜色, marker:点的形状
ax5.scatter(df5['humidity'], df5['count'], s=df5['count']/10, c=df5['count'], marker='.')
ax5.set_title('2011-2012 bike sharing demand by humidity')
ax5.set_xlabel('humidity')
ax5.set_ylabel('count')

ax6 = plt.subplot(1, 2, 2)
df6 = climateDf[['temp', 'count']]
ax6.scatter(df6['temp'], df6['count'], s=df6['count']/10, c=df6['count'], marker='.')
ax6.set_title('2011-2012 bike sharing demand by temperature')
ax6.set_xlabel('temperature')
ax6.set_ylabel('count')
plt.savefig(r'./散点图.png', dpi=400)

# 显示所有的图
plt.show()

