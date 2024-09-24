import logging  

def setup_logging(filename='train.log', level=logging.INFO):  
    """  
    配置logging，用于记录日志到指定的文件  
  
    :param filename: 日志文件的名称  
    """  
    logging.basicConfig(filename=filename,  
                        level=level,  
                        format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s',  
                        datefmt='%Y-%m-%d %H:%M:%S')  # 可以添加datefmt来定义时间格式
  

