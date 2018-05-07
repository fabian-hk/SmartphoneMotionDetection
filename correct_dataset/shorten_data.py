import json

file_name = "1_class_time_000005.txt"
shorten = 3

file = open(file_name, "r")

array = json.loads(file.readline())

ending = array.__len__() - shorten

newArray = array[0:ending]

print("old lenth: "+str(array.__len__()))
print("new length: "+str(newArray.__len__()))

saveFile = open("new_"+file_name, "w")
saveFile.write(json.dumps(newArray))

