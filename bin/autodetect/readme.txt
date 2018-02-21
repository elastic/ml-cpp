1. Run the program

./.objs/autodetect --fieldconfig=/Users/dkyle/Data/farequote.conf --timefield=time --delimiter=',' --timeformat='%F %T'  < ~/Data/farequote.csv

./.objs/autodetect --fieldconfig=/Users/dkyle/Data/farequote_ppn.conf --timefield=time --delimiter=',' < ~/Data/farequote_epoch.csv

.objs/autodetect --fieldconfig=firewall.conf --delimiter=',' --bucketspan=3600 < /Volumes/test_data/influence/security_story1/firewall.log


2. With time field

.objs/autodetect --fieldconfig=farequote.conf --delimiter=',' --timefield=time --bucketspan=3600 --timeformat='%F %T' < farequote_ISO_8601.csv

.objs/autodetect --fieldconfig=/Users/dkyle/tmp/sm.conf --delimiter=',' --timefield=time --bucketspan=600 --timeformat='%d/%b/%Y %T' < /Volumes/test_data/engine_api_integration_test/rare/r03a-data.txt


3. FlightCentre

.objs/autodetect --fieldconfig=.objs/flightcentre.conf --delimiter=',' --bucketspan=3600 < .objs/flightcentre_forwards.csv



4. Restore state

# process first batch of data
.objs/autodetect --fieldconfig=.objs/flightcentre.conf --delimiter=',' --bucketspan=3600 --persistState < .objs/flightcentre_forwards_1.csv

# restore state and process the 2nd batch of data
.objs/autodetect --fieldconfig=.objs/flightcentre.conf --delimiter=',' --bucketspan=3600 --restoreState=/tmp/model_state < .objs/flightcentre_forwards_2.csv

--
.objs/autodetect --fieldconfig=.objs/flightcentre.conf --delimiter=',' --bucketspan=3600 --restoreState=/Users/dkyle/tmp/cmdline_model_state/model_state < .objs/flightcentre_forwards_2.csv

.objs/autodetect --fieldconfig=.objs/flightcentre.conf --delimiter=',' --bucketspan=3600 --restoreState=/Users/dkyle/tmp/engine_model_state/model_state < .objs/flightcentre_forwards_2.csv


86400 = 1 day
3600 = 1 hour
