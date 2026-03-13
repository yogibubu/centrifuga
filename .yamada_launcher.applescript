on run
  set pybin to quoted form of "/opt/miniconda3/bin/python3"
  set pysrc to quoted form of "/Users/vincenzobarone/centrifugal/yamada_gui.py"
  set cmd to pybin & " " & pysrc & " >/tmp/yamada_gui_app.log 2>&1 &"
  do shell script cmd
end run
