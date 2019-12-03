from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

def demo1():
    # 新建地图
    map = Basemap() #Basemap类有很多属性，这里全都使用默认参数

    # 画图
    map.drawcoastlines()

    # 显示结果
    plt.show()

    # 存储结果
    plt.savefig('./test.png')

def unit_test():
    demo1()

if __name__ == '__main__':
    unit_test()