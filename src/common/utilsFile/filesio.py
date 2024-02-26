import os



class FilesIO:

    """
    这是一个文件IO流类
    用于处理文件读取以及文件路径获取
    """

    @staticmethod
    def getRootPath():

        """
        获取项目根目录
        """

        common_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        src_path = os.path.dirname(common_path)
        root_path = os.path.dirname(src_path)
        return root_path
    
    
    @staticmethod
    def getDataPath(file_name: str):

        """
        获取数据集路径
        """

        root_path = FilesIO.getRootPath()
        resources_path = os.path.join(root_path, 'resources')
        data_path = os.path.join(resources_path, file_name)
        return data_path
