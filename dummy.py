import os


utterance_folder = os.listdir("/app/BS/utterances")

files = [f for f in utterance_folder if f.endswith(".csv")]

for file in files:
    #put the elements inside " "
    with open("/app/BS/utterances/" + file, "r") as f:
        lines = f.readlines()
    with open("/app/BS/utterances/" + file, "w") as f:
        for line in lines:
            # Strip trailing newline before quoting to avoid embedded line breaks
            f.write('"' + line.rstrip("\n") + '"\n')

test_folder = os.listdir("/app/BS/test_suites")

files = [f for f in test_folder if f.endswith(".csv")]

for file in files:
    #put the elements inside " "
    with open("/app/BS/test_suites/" + file, "r") as f:
        lines = f.readlines()
    with open("/app/BS/test_suites/" + file, "w") as f:
        for line in lines:
            # Strip trailing newline before quoting to avoid embedded line breaks
            f.write('"' + line.rstrip("\n") + '"\n')

print("Done")
