import json
import sys
import sh


def jq_reformat(input):
    "Reformat the almost-but-not-quite JSON input to remove duplicate object names to make it acceptable to the json parser"
    json_string = sh.jq('-cn', '--stream', 'def fromstream_with_dups(i):\
  foreach i as $i (\
    [null, null];\
\
    if ($i | length) == 2 then\
      if ($i[0] | length) == 0 then .\
      elif $i[0][-1]|type == "string" then\
        [ ( .[0] | setpath($i[0]; getpath($i[0]) + [$i[1]]) ), .[1] ]\
      else [ ( .[0] | setpath($i[0]; $i[1]) ), .[1] ]\
      end\
    elif ($i[0] | length) == 1 then [ null, .[0] ]\
    else .\
    end;\
\
    if ($i | length) == 1 then\
      if ($i[0] | length) == 1 then .[1]\
      else empty\
      end\
    elif ($i[0] | length) == 0 then $i[1]\
    else empty\
    end\
  );\
  fromstream_with_dups(inputs)', _in=input, _tty_out=False)
    return str(json_string)

def parse_model_state_json(json_string):
    try:
        obj = json.loads(json_string)
        if 'index'  in obj:
            print("Residual data for index id {}".format(obj['index']['_id'][0]))
        elif 'residual_model' in obj:
            priors = obj['residual_model']['one-of-n']['model']['prior']
            for name, prior in priors.items():
                if name != 'multimodal':
                    if prior['mean'][0] == '<unknown>' or prior['standard_deviation'][0] == '<unknown>':
                        continue
                    mean = float(prior['mean'][0])
                    sd = float(prior['standard_deviation'][0])
                    print("{name}: mean = {mean:f}, sd = {sd:f}".format(name=name, mean=mean, sd=sd))
        else:
            pass
    except:
        sys.exit("Error: Cannot parse JSON document '" + str(json_string) + "' Encountered " + str(sys.exc_info()[0]))

    return

if __name__ == '__main__':

    data=''
    if len(sys.argv) < 2:
        data = sys.stdin.read();
    else:
        fileName = sys.argv[1]

        try:
            jsonFile = open(fileName, 'r')
        except IOError:
            sys.exit("Error: Cannot open file '" + fileName)

        print ("Reading items from {}".format(fileName))

        with open(fileName) as json_file:  
            data=json_file.read()

    reformatted_json = jq_reformat(data)

    for line in reformatted_json.splitlines():
        if line != '0':
            parse_model_state_json(line)


