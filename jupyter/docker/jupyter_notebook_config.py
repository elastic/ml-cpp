# Configuration file for jupyter-notebook.

## Whether to allow the user to run the notebook as root.
c.NotebookApp.allow_root = True

## The full path to an SSL/TLS certificate file.
c.NotebookApp.certfile = u'/root/mycert.pem'


## The IP address the notebook server will listen on.
c.NotebookApp.ip = '*'

## The full path to a private key file for usage with SSL/TLS.
c.NotebookApp.keyfile = u'/root/mykey.key'


## Whether to open in a browser after starting. The specific browser used is
#  platform dependent and determined by the python standard library `webbrowser`
#  module, unless it is overridden using the --browser (NotebookApp.browser)
#  configuration option.
c.NotebookApp.open_browser = False

## Hashed password to use for web authentication.
#  
#  To generate, type in a python/IPython shell:
#  
#    from notebook.auth import passwd; passwd()
#  
#  The string should be of the form type:salt:hashed-password.
c.NotebookApp.password=u'argon2:$argon2id$v=19$m=10240,t=10,p=8$CoS95BFTAOIVlBhwFE2j2g$jnbxwQxlCJePqYTX1Bakrw'

## The port the notebook server will listen on (env: JUPYTER_PORT).
c.NotebookApp.port = 9999
