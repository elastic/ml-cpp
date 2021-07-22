from pathlib2 import Path
import platform
from sys import exit

root_dir = Path(__file__).parent.parent.parent.absolute()
data_dir = root_dir / 'data'
datasets_dir = data_dir / 'datasets'
configs_dir = data_dir / 'configs'
dfa_path = ''

# I assume, your host OS is not CentOS
cloud = (platform.system() == 'Linux') and (platform.dist()[0] == 'centos')
if cloud:
    search_location = Path('/ml-cpp')
    dfa_path = Path('/ml-cpp/bin/data_frame_analyzer')
else:
    search_location = Path(__file__).parent.parent.parent.parent
    runners = list(search_location.glob('**/build/distribution/platform/**/data_frame_analyzer'))
    if len(runners) == 0:
        print("Cannot find data_frame_analyzer binary")
        exit(1)
    dfa_path = runners[0]
