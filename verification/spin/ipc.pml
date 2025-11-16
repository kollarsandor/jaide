/* JAIDE v40 Spin Model: IPC Deadlock Freedom */

#define NCLIENTS 4
#define BUFSIZE 8

typedef message {
  byte sender;
  byte data;
}

chan channels[NCLIENTS] = [BUFSIZE] of { message };
bool waiting[NCLIENTS];
bool active[NCLIENTS];

byte send_cap[NCLIENTS];
byte recv_cap[NCLIENTS];

init {
  byte i;
  atomic {
    i = 0;
    do
    :: i < NCLIENTS ->
       active[i] = 1;
       send_cap[i] = 1;
       recv_cap[i] = 1;
       waiting[i] = 0;
       i++
    :: i >= NCLIENTS -> break
    od
  }
}

proctype Client(byte id) {
  message msg;
  byte target;
  
  do
  :: active[id] && send_cap[id] ->
     atomic {
       select(target : 0 .. (NCLIENTS-1));
       if
       :: target != id && len(channels[target]) < BUFSIZE ->
          msg.sender = id;
          msg.data = id * 10 + target;
          channels[target]!msg;
          waiting[target] = 0
       :: else -> skip
       fi
     }
     
  :: active[id] && recv_cap[id] ->
     atomic {
       if
       :: nempty(channels[id]) ->
          channels[id]?msg;
          waiting[id] = 0
       :: empty(channels[id]) ->
          waiting[id] = 1
       fi
     }
     
  :: active[id] ->
     skip  /* Idle step */
  od
}

active proctype Monitor() {
  assert(!(waiting[0] && waiting[1] && waiting[2] && waiting[3]));
}

never {
  do
  :: skip
  :: (waiting[0] && waiting[1] && waiting[2] && waiting[3]) -> goto accept
  od;
accept:
  do
  :: (waiting[0] && waiting[1] && waiting[2] && waiting[3]) -> skip
  od
}

ltl no_deadlock { []!((waiting[0] && waiting[1] && waiting[2] && waiting[3])) }

ltl message_delivered {
  [](nempty(channels[0]) -> <>(empty(channels[0])))
}

ltl no_starvation {
  []<>(!waiting[0]) && []<>(!waiting[1]) && 
  []<>(!waiting[2]) && []<>(!waiting[3])
}

active proctype Launcher() {
  byte i = 0;
  do
  :: i < NCLIENTS ->
     run Client(i);
     i++
  :: i >= NCLIENTS -> break
  od
}
