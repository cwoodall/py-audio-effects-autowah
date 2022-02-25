# autowah

![](docs/images/screenshot-of-gui.png)

## Installing

### Linux

You will need the port audio C library installed with header files, on Ubuntu 20.04 based distributions the following works:

```
sudo apt install portaudio19-dev
```

## Running 

```
poetry run autowah &; chrt -b --pid $!
```

```mermaid
graph LR
    Audio --> EnvelopeDetector
    Audio --> VariableBandwidthLPF
    EnvelopeDetector --> VariableBandwidthLPF  --> AudioOut
```

### Todo

#### Done
- [x] Cleanup Envelope Detector implementation
- [x] Implement variable bandwidth filter (needs to be computed per sample)
- [x] Cleanup the variable bw filter
- [X] Tuneable Q filter (IIR)
- [x] Fix the underruns based on the filter len... Or get a better tuneable Q filter in there...
- [x] Finish the autowah integration
- [x] control variables and interactive plot
- [x] Get CI working

#### Important before Blog Post

- [ ] Add a compression block
- [ ] Apply to a wav file
- [ ] Documentation sweep of the code base
- [ ] Breakout unit tests for each section

#### Blog Post

- [ ] Record some videos
- [ ] Abstract block type + wiring diagram concept?
- [ ] Create plots and gather resources
- [ ] Plots of the movable filter. Maybe even some interactive graphics?
- [ ] Diagram out architecture of the system
- [ ] Write blog post
- [ ] Publish

#### Someday
- [ ] Fix exiting the program cleanly
- [ ] Migrate to a pyqtplot based UI
- [ ] Clean up the scope code
