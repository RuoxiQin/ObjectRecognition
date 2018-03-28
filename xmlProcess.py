import xml.etree.ElementTree as ET

tree = ET.parse('/Users/haohui/Desktop/scene.xml')
root = tree.getroot()

for object in root.iter('object'):
	if(object.find('deleted').text == "0"):
		object_name = object.find('name').text
		print("object_name: "+object_name)
		object_attrib = object.find('attributes').text
		print("object_attrib: "+object_attrib)

		x_list = []
		y_list = []
		
		for pt in object.iter('pt'):
			x = pt.find('x').text
			y = pt.find('y').text
			x_list.append(x)
			y_list.append(y)
		
		x_list.sort()
		y_list.sort()

		print("x_min:"+x_list[0], "x_max:"+x_list[-1], "y_min:"+y_list[0], "y_max:"+y_list[-1])
		print('\n')
	else: 
		continue

