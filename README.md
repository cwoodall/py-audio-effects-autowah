# autowah


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

- [ ] Cleanup Envelope Detector implementation
- [ ] Implement variable bandwidth filter (needs to be computed per sample)
- [ ] Create plots and gather resources
- [ ] Diagram out architecture of the system
- [ ] Write blog post
- [ ] Publish