'''
 决策树的 处理模块
	1 retrieveTree  检索整棵树，组装成字典
	2 createPlot    创建图像
'''
def retrieveTree(i):
	listOfTrees = 	[
						{
							'no surfacing':{
								0: 'no',
								1:{
									'flippers':{
										0:'no',
										1:'yes'
									}
								}
							}
						},
						{
							'no surfacing':{
								0:'no',
								1:{
									'flippers':{
										0:{
											'head':{
												0:'no',
												1:'yes'
											}
										},
										1:'no'
									}
								}
							}
						}
					
					]
	return listOfTrees[i]
	
