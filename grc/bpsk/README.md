This is a recreation of Barry Duggan's work. His files and excellent
explanations are available here:

* https://github.com/duggabe/gr-control
* https://wiki.gnuradio.org/index.php?title=File_transfer_using_Packet_and_BPSK

To test this using sockets, without using RF:

1. Clone this repository.

2. Clone https://github.com/duggabe/gr-control.

3. Open several radioconda terminals and GNU Radio flowgraphs. Be sure to
   launch the transmitter last:

   - This directory.
     - `python -u bpsk_pkt_rcv.py`
   - duggabe/gr-control
     - `python -u chan_loopback.py`
   - duggabe/gr-control/Transmitters
     - `python -u pkt_xmy.py --InFile="..\gr-logo.png"`

4. When the receive flowgraph stops updating, reception is finished. Close the
   transmit and receive flowgraph windows.

5. Strip the preamble: `python strip_preamble.py output.tmp output.bin`

6. If successful, output will show `End of text` _and_ `Transmitted file name`.
