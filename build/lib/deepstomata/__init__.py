from . import utils
import sys, os, time, statistics

name = "deepstomata"

def deepstomata(dir_path, config_path = os.path.dirname(__file__)+"/config.ini"):
    #silence deprecation warning
    import sys
    import warnings

    if not sys.warnoptions:
        warnings.simplefilter("ignore")



    print ("Reading config file")
    if config_path == os.path.join(os.path.dirname(__file__), "config.ini"):
        print ("using default config.ini")
    else:
        print ("using user defined config.ini from:", config_path)
    utils.import_config(config_path)
    print("config file imported properly")

    print("listing image files in:", dir_path)
    item_list = utils.check_type_of_input(dir_path)
    print ("Will be analyzing the listed files...")
    print([os.path.basename(x) for x in item_list])


    print ("analysis start")
    time_container = []
    all_start = time.time()

    #main process
    for item in item_list:
        print (" ")
        print (os.path.basename(item))
        start = time.time()
        utils.analyze(item)  # core module
        end = time.time()
        time_container.append(end - start)

    all_end = time.time()

    print ("Finished. csv files and annotated images are generated in the input directory. \n")
    with open("time.txt","w") as f:
        if len(time_container) > 1:
            s = statistics.mean(time_container)
            s2 = statistics.stdev(time_container)
            print ("mean time processing:" , s)
            print ("stdev time processing:" , s2)
            f.write("mean"+str(s)+"\n")
            f.write("stdev"+str(s2)+"\n")
        f.write("total"+str(all_end - all_start)+"\n")
    print ("total time:", all_end - all_start)
