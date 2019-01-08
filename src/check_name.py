import os

pic_list = os.listdir("./src/trainImage_4")
print(len(pic_list))

wrong_ = "~!@#$^&*()/"

for name in pic_list:
    for i_str in wrong_:
        if i_str in name:
            print("wrong name!--> ", name)
            new_name = input("new name: ")
            os.rename(name, new_name)