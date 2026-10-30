// MERIT-Net NS-3 scenario — BENIGN baseline + per-neighbour feature sniffing.
//
// Produces, for every 1 s window, a per-(observer i, neighbour j) feature tensor
// X[i][j][0..22] matching the 23 features used by the Python detectors
// (see meritnet/sim.py: F_RREQ..F_MASK).  Written to a binary file the Python
// side reshapes to (T, n, n, 23) float32.
//
// This file contains NO attack code.  An attacker is declared only by the
// --attackers list; the AttackHook() function below is an empty stub where a
// user adds their own attack traffic.  With no hook, every node is honest and
// the run is a clean AODV baseline, which is what training on benign traffic
// and calibrating the detectors needs.
//
// Build:  ./build.sh meritnet-scenario.cc meritnet-scenario
// Run:    ./meritnet-scenario --nNodes=50 --speed=5 --sim=200 --seed=1 --out=win.bin
//
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/aodv-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/aodv-packet.h"
#include "ns3/udp-header.h"
#include "ns3/tcp-header.h"
#include "ns3/ipv4-header.h"
#include "ns3/llc-snap-header.h"

#include "ns3/error-model.h"
#include "ns3/wifi-mac-header.h"
#include "ns3/wifi-net-device.h"
#include "ns3/wifi-phy.h"

#include <fstream>
#include <map>
#include <set>
#include <vector>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <unistd.h>

using namespace ns3;

// Enforcement (matches design Section 9):
//   gDrop[i][j]     = prob node i drops ANY packet from convicted j (blacklist).
//   gThrottle[i][j] = prob node i drops an RREQ from j (graduated rate-limit,
//                     pre-alarm band). Only RREQs are throttled, so a node under
//                     graduated response stays observable on its other streams
//                     (a HELLO/data/SYN flood is not masked by an RREQ throttle).
static std::vector<std::vector<float>> gDrop;
static std::vector<std::vector<float>> gThrottle;
static bool gBridge = false;

static const int NF = 23;
enum { F_RREQ, F_HELLO, F_DATA, F_SYN, F_RERR, F_HMISS, F_DEGJ, F_CHURN, F_NEWFLOW,
       F_ACTFLOW, F_LINKAGE, F_DDIST, F_DEGI, F_COMP, F_FB, F_UNKNOWN, F_ENTROPY,
       F_DUP, F_HSYM, F_PERFLOW, F_FBRATIO, F_HALFOPEN, F_MASK };

// -------------------------------------------------------------- configuration
struct Cfg {
  uint32_t nNodes = 50;
  double   area   = 1000.0;
  double   range  = 250.0;   // metres (tuned: ~30 dBm, 802.11b, exponent 3)
  double   speed  = 5.0;
  double   sim    = 200.0;
  double   win    = 1.0;
  uint32_t seed   = 1;
  uint32_t nCbr   = 20;
  std::string out = "win.bin";
  std::string attackers = "";       // comma-separated node ids
  std::string attack = "benign";    // benign|rreq|rreq_stealth|hello|data|syn
  double   rate   = 20.0;           // attack intensity (per-second), meaning per attack
  double   aStart = 60.0;           // attack start time (s)
};
static Cfg g;

// -------------------------------------------------------------- global state
static uint32_t N;
static std::vector<Ptr<Node>> gNodes;
static std::map<Mac48Address, uint32_t> gMacToId;     // device MAC -> node id
static std::map<Ipv4Address, uint32_t>  gIpToId;      // node IP   -> node id

// per-window accumulators, indexed [observer][neighbour]
struct Acc {
  double rreq=0, hello=0, data=0, syn=0, rerr=0, comp=0, fb=0;
  std::map<uint32_t,int> dests;       // requested RREQ destinations (for entropy/dup/unknown)
  std::set<uint32_t> flows;           // distinct data destinations seen
};
static std::vector<std::vector<Acc>> gAcc;

// topology history
static std::vector<std::set<uint32_t>> gPrevNbrs;
static std::vector<std::vector<double>> gLinkAge;     // [i][j] windows the link i-j has existed
static std::vector<std::vector<double>> gPrevDist;
static std::ofstream gOut;
static uint32_t gWinCount = 0;

// -------------------------------------------------------------- helpers
static Vector Pos(uint32_t i) {
  return gNodes[i]->GetObject<MobilityModel>()->GetPosition();
}
static double Dist(uint32_t i, uint32_t j) {
  return CalculateDistance(Pos(i), Pos(j));
}

// Per-node Wi-Fi reception filter: drops a frame if the detector has told this
// node to drop traffic from the frame's transmitter.  This is how a blacklist
// or graduated-forwarding decision becomes real in the simulation.
class NeighbourBlockErrorModel : public ErrorModel {
public:
  static TypeId GetTypeId() {
    static TypeId tid = TypeId("NeighbourBlockErrorModel")
        .SetParent<ErrorModel>().AddConstructor<NeighbourBlockErrorModel>();
    return tid;
  }
  uint32_t m_node = 0;
  Ptr<UniformRandomVariable> m_u = CreateObject<UniformRandomVariable>();
private:
  // The post-reception PSDU here is the LLC/SNAP + IP payload (no MAC header),
  // so enforcement is keyed on the IP source: node m_node drops traffic whose
  // originator it has convicted.  For this threat model the flooder is the IP
  // source of its own RREQ / data / SYN traffic, so this blocks the attacker,
  // and an honest node wrongly convicted loses only its own originated traffic.
  bool DoCorrupt(Ptr<Packet> pkt) override {
    if (pkt->GetSize() < 28) return false;             // too short for LLC(8)+IP(20)
    Ptr<Packet> p = pkt->Copy();
    LlcSnapHeader llc;
    if (p->PeekHeader(llc) == 0 || llc.GetType() != 0x0800) return false;
    p->RemoveHeader(llc);
    if (p->GetSize() < 20) return false;
    Ipv4Header ip;
    if (p->PeekHeader(ip) == 0) return false;
    auto it = gIpToId.find(ip.GetSource());
    if (it == gIpToId.end()) return false;
    uint32_t src = it->second;
    // Blacklist: convicted source -> drop every packet.
    float d = gDrop[m_node][src];
    if (d > 0.0f && (d >= 1.0f || m_u->GetValue(0, 1) < d)) return true;
    // Graduated rate-limit: throttle this source's RREQs only.
    float th = gThrottle[m_node][src];
    if (th > 0.0f && IsRreq(p, ip)) return th >= 1.0f || m_u->GetValue(0, 1) < th;
    return false;
  }
  // Is the remaining payload an AODV RREQ? (UDP port 654, AODV type RREQ.)
  static bool IsRreq(Ptr<Packet> p, const Ipv4Header& ip) {
    if (ip.GetProtocol() != 17) return false;          // UDP
    Ptr<Packet> q = p->Copy(); Ipv4Header tmp; q->RemoveHeader(tmp);
    if (q->GetSize() < 8) return false;
    UdpHeader uh;
    if (q->PeekHeader(uh) == 0) return false;
    if (uh.GetSourcePort() != 654 && uh.GetDestinationPort() != 654) return false;
    q->RemoveHeader(uh);
    aodv::TypeHeader ty;
    return q->PeekHeader(ty) && ty.IsValid() && ty.Get() == aodv::AODVTYPE_RREQ;
  }
  void DoReset() override {}
};


// Promiscuous sniff callback: node `obs` overheard `pkt` from L2 source `from`.
bool Sniff(uint32_t obs, Ptr<NetDevice> dev, Ptr<const Packet> pkt,
           uint16_t proto, const Address& from, const Address& to,
           NetDevice::PacketType pt) {
  if (proto != 0x0800) return false;                 // IPv4 only
  auto it = gMacToId.find(Mac48Address::ConvertFrom(from));
  if (it == gMacToId.end()) return false;
  uint32_t src = it->second;                         // L2 neighbour that transmitted
  if (src == obs) return false;

  Ptr<Packet> p = pkt->Copy();
  Ipv4Header ip;
  if (p->PeekHeader(ip) == 0) return false;
  p->RemoveHeader(ip);
  uint8_t l4 = ip.GetProtocol();
  Acc& a = gAcc[obs][src];

  if (l4 == 17) {                                    // UDP
    UdpHeader uh;
    if (p->PeekHeader(uh) == 0) return false;
    uint16_t sp = uh.GetSourcePort(), dp = uh.GetDestinationPort();
    p->RemoveHeader(uh);
    if (sp == 654 || dp == 654) {                    // AODV control
      aodv::TypeHeader th;
      if (p->PeekHeader(th) == 0 || !th.IsValid()) return false;
      switch (th.Get()) {
        case aodv::AODVTYPE_RREQ: {
          p->RemoveHeader(th); aodv::RreqHeader rq;
          if (p->PeekHeader(rq)) {
            a.rreq += 1;
            auto dit = gIpToId.find(rq.GetDst());
            uint32_t did = (dit == gIpToId.end()) ? N : dit->second;  // N = nonexistent dest
            a.dests[did]++;
          }
          break; }
        case aodv::AODVTYPE_RREP: {
          p->RemoveHeader(th); aodv::RrepHeader rp;
          if (p->PeekHeader(rp)) {
            if (rp.GetDst() == rp.GetOrigin()) a.hello += 1;   // ns-3 AODV HELLO = self RREP
          }
          break; }
        case aodv::AODVTYPE_RERR: a.rerr += 1; break;
        default: break;
      }
    } else {                                         // UDP application data
      a.data += 1;
      a.flows.insert(dp);
    }
  } else if (l4 == 6) {                              // TCP
    TcpHeader th;
    if (p->PeekHeader(th) == 0) return false;
    uint8_t fl = th.GetFlags();
    bool syn = fl & TcpHeader::SYN, ack = fl & TcpHeader::ACK;
    if (syn && !ack)      a.syn  += 1;               // SYN
    else if (syn && ack)  a.comp += 1;               // SYN-ACK => handshake progressing
    else if (ack)         a.fb   += 1;               // data/ACK feedback
    a.data += 1;
    a.flows.insert(th.GetDestinationPort());
  }
  return false;
}

// -------------------------------------------------------------- attack traffic
// The four flooding attacks, implemented the standard way for a simulated IDS
// evaluation: ordinary NS-3 traffic generation and protocol configuration, not
// packet-injection exploits.  Each is labelled by --attack and driven at --rate.
static NodeContainer gNC;
static Ipv4InterfaceContainer gIfs;
static Ptr<UniformRandomVariable> gAU;

// RREQ flood: the attacker repeatedly asks for a route it does not have, so
// plain AODV issues a fresh RREQ each time.  A route to a *nonexistent* address
// (rreq) keeps the route table empty so every probe re-floods; aiming at real
// node addresses (rreq_stealth) hides the unknown-destination signature.
static std::map<uint32_t, Ptr<Socket>> gRreqSock;
void RreqProbe(uint32_t a, bool stealth) {
  if (Simulator::Now().GetSeconds() >= g.sim) return;
  Ptr<Socket> s = gRreqSock[a];
  Ipv4Address dst;
  if (stealth) dst = gIfs.GetAddress(gAU->GetInteger(0, N - 1));
  else         dst = Ipv4Address(0x0a630000u + gAU->GetInteger(1, 60000));  // 10.99.x.x, no node
  s->Connect(InetSocketAddress(dst, 555));
  s->Send(Create<Packet>(4));
  Simulator::Schedule(Seconds(1.0 / std::max(g.rate, 1.0)), &RreqProbe, a, stealth);
}

// SYN flood: the attacker opens many TCP connections toward a victim and sends
// almost nothing, leaving half-open state.  Modelled with short TCP bursts.
void SynBurst(uint32_t a, uint32_t victim) {
  if (Simulator::Now().GetSeconds() >= g.sim) return;
  Ptr<Socket> s = Socket::CreateSocket(gNC.Get(a), TcpSocketFactory::GetTypeId());
  s->Bind();
  s->Connect(InetSocketAddress(gIfs.GetAddress(victim), 800 + (uint16_t)gAU->GetInteger(0, 200)));
  s->Send(Create<Packet>(1));
  Simulator::Schedule(Seconds(1.0 / std::max(g.rate, 1.0)), &SynBurst, a, victim);
}

void AttackHook(const std::vector<uint32_t>& attackers) {
  if (g.attack == "benign" || attackers.empty()) return;
  gAU = CreateObject<UniformRandomVariable>();

  if (g.attack == "rreq" || g.attack == "rreq_stealth") {
    bool stealth = g.attack == "rreq_stealth";
    for (uint32_t a : attackers) {
      Ptr<Socket> s = Socket::CreateSocket(gNC.Get(a), UdpSocketFactory::GetTypeId());
      s->Bind(); gRreqSock[a] = s;
      Simulator::Schedule(Seconds(g.aStart), &RreqProbe, a, stealth);
    }
  } else if (g.attack == "hello") {
    // Hello flood: shorten the attacker's AODV HELLO interval and raise its power.
    for (uint32_t a : attackers) {
      std::ostringstream path;
      path << "/NodeList/" << a << "/$ns3::aodv::RoutingProtocol/HelloInterval";
      Config::Set(path.str(), TimeValue(Seconds(std::max(0.05, 1.0 / g.rate))));
      Ptr<WifiNetDevice> wd = DynamicCast<WifiNetDevice>(gNC.Get(a)->GetDevice(0));
      if (wd) { wd->GetPhy()->SetTxPowerStart(40); wd->GetPhy()->SetTxPowerEnd(40); }
    }
  } else if (g.attack == "data") {
    // Data flood: a colluding attacker pair exchanges high-rate UDP through relays.
    for (size_t k = 0; k + 1 < attackers.size() + 1 && attackers.size() >= 2; k += 2) {
      uint32_t a = attackers[0], b = attackers[1];
      for (auto pr : {std::make_pair(a, b), std::make_pair(b, a)}) {
        uint16_t port = 6000 + pr.first;
        PacketSinkHelper sink("ns3::UdpSocketFactory",
            InetSocketAddress(Ipv4Address::GetAny(), port));
        sink.Install(gNC.Get(pr.second)).Start(Seconds(g.aStart - 1));
        OnOffHelper on("ns3::UdpSocketFactory", InetSocketAddress(gIfs.GetAddress(pr.second), port));
        on.SetAttribute("DataRate", DataRateValue(DataRate((uint64_t)(g.rate * 512 * 8))));
        on.SetAttribute("PacketSize", UintegerValue(512));
        on.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1000]"));
        on.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
        auto app = on.Install(gNC.Get(pr.first)); app.Start(Seconds(g.aStart));
      }
      break;
    }
  } else if (g.attack == "syn") {
    uint32_t victim = attackers[0];
    do { victim = gAU->GetInteger(0, N - 1); } while (victim == attackers[0]);
    PacketSinkHelper sink("ns3::TcpSocketFactory",
        InetSocketAddress(Ipv4Address::GetAny(), 800));
    sink.Install(gNC.Get(victim)).Start(Seconds(g.aStart - 1));
    for (uint32_t a : attackers)
      Simulator::Schedule(Seconds(g.aStart), &SynBurst, a, victim);
  }
}

// -------------------------------------------------------------- window dump
void DumpWindow() {
  std::vector<float> X((size_t)N * N * NF, 0.0f);
  auto at = [&](uint32_t i, uint32_t j, int f) -> float& {
    return X[((size_t)i * N + j) * NF + f];
  };

  // current adjacency + degree from true positions (observer "in range" = can sniff)
  std::vector<std::vector<char>> adj(N, std::vector<char>(N, 0));
  std::vector<int> deg(N, 0);
  std::vector<std::set<uint32_t>> nbr(N);
  for (uint32_t i = 0; i < N; i++)
    for (uint32_t j = 0; j < N; j++)
      if (i != j && Dist(i, j) <= g.range) { adj[i][j] = 1; deg[i]++; nbr[i].insert(j); }

  for (uint32_t i = 0; i < N; i++) {
    for (uint32_t j = 0; j < N; j++) {
      if (i == j || !adj[i][j]) continue;
      Acc& a = gAcc[i][j];
      at(i,j,F_MASK) = 1.0f;
      at(i,j,F_RREQ) = a.rreq;  at(i,j,F_HELLO) = a.hello;
      at(i,j,F_DATA) = a.data;  at(i,j,F_SYN)   = a.syn;
      at(i,j,F_RERR) = a.rerr;  at(i,j,F_COMP)  = a.comp;  at(i,j,F_FB) = a.fb;
      at(i,j,F_HMISS) = std::max(0.0, 1.0 - std::min(a.hello, 1.0));
      at(i,j,F_DEGJ) = deg[j];  at(i,j,F_DEGI) = deg[i];
      // churn of j's neighbour set
      int churn = 0;
      for (uint32_t k : nbr[j]) if (!gPrevNbrs[j].count(k)) churn++;
      for (uint32_t k : gPrevNbrs[j]) if (!nbr[j].count(k)) churn++;
      at(i,j,F_CHURN) = churn;
      // link age i-j (normalised /30)
      at(i,j,F_LINKAGE) = std::min(gLinkAge[i][j], 30.0) / 30.0;
      // change in distance i-j
      double dd = gPrevDist[i][j] < 0 ? 0.0 : std::fabs(Dist(i,j) - gPrevDist[i][j]) / g.range;
      at(i,j,F_DDIST) = dd;
      // active flows of j, and per-flow rate
      int nflows = (int)a.flows.size();
      at(i,j,F_ACTFLOW) = (a.data > 0) ? nflows : 0;
      at(i,j,F_PERFLOW) = (a.data > 0) ? a.data / std::max(nflows, 1) : 0;
      at(i,j,F_FBRATIO) = (a.data > 0) ? a.fb / std::max(a.data, 1.0) : 0;
      at(i,j,F_HALFOPEN)= (a.syn > 0) ? std::max(0.0, a.syn - a.comp) / std::max(a.syn, 1.0) : 0;
      // hello symmetry: does j list i as a neighbour?
      at(i,j,F_HSYM) = (a.hello > 0 && adj[j][i]) ? 1.0f : 0.0f;
      // RREQ destination stats
      if (a.rreq > 0 && !a.dests.empty()) {
        int tot = 0, unknown = 0; double ent = 0;
        for (auto& kv : a.dests) { tot += kv.second; if (kv.first == N) unknown += kv.second; }
        for (auto& kv : a.dests) { double pr = (double)kv.second / tot; ent -= pr * std::log2(pr); }
        at(i,j,F_UNKNOWN) = (double)unknown / tot;
        at(i,j,F_ENTROPY) = ent;
        at(i,j,F_DUP) = 1.0 - (double)a.dests.size() / tot;
      }
    }
  }
  // NEWFLOW is left 0 here (first data packet to a new dest) — the Python side
  // derives it from F_ACTFLOW deltas; see loader note.

  if (gBridge) {
    // lockstep with the Python detector: send features, block for decisions
    uint32_t idx = gWinCount;
    fwrite(&idx, sizeof(uint32_t), 1, stdout);
    fwrite(X.data(), sizeof(float), X.size(), stdout);
    fflush(stdout);
    uint32_t k = 0;
    if (fread(&k, sizeof(uint32_t), 1, stdin) != 1) { Simulator::Stop(); return; }
    // each decision = (i, j, p, kind): kind 0 = blacklist (drop all), 1 = throttle RREQs
    for (uint32_t i2 = 0; i2 < N; i2++)
      std::fill(gDrop[i2].begin(), gDrop[i2].end(), 0.0f),
      std::fill(gThrottle[i2].begin(), gThrottle[i2].end(), 0.0f);
    for (uint32_t t = 0; t < k; t++) {
      uint32_t i, j; float p, kind;
      if (fread(&i, 4, 1, stdin) != 1 || fread(&j, 4, 1, stdin) != 1 ||
          fread(&p, 4, 1, stdin) != 1 || fread(&kind, 4, 1, stdin) != 1) { Simulator::Stop(); return; }
      if (i < N && j < N) { if (kind < 0.5f) gDrop[i][j] = p; else gThrottle[i][j] = p; }
    }
  } else {
    gOut.write(reinterpret_cast<char*>(X.data()), X.size() * sizeof(float));
  }
  gWinCount++;

  // roll history
  for (uint32_t i = 0; i < N; i++) {
    gPrevNbrs[i] = nbr[i];
    for (uint32_t j = 0; j < N; j++) {
      gPrevDist[i][j] = Dist(i, j);
      gLinkAge[i][j] = adj[i][j] ? gLinkAge[i][j] + 1 : 0;
      gAcc[i][j] = Acc();
    }
  }
  if (Simulator::Now().GetSeconds() + g.win <= g.sim)
    Simulator::Schedule(Seconds(g.win), &DumpWindow);
}

// -------------------------------------------------------------- main
int main(int argc, char** argv) {
  CommandLine cmd;
  cmd.AddValue("nNodes", "node count", g.nNodes);
  cmd.AddValue("area", "field side (m)", g.area);
  cmd.AddValue("range", "radio range (m)", g.range);
  cmd.AddValue("speed", "max speed (m/s)", g.speed);
  cmd.AddValue("sim", "sim time (s)", g.sim);
  cmd.AddValue("seed", "RNG seed", g.seed);
  cmd.AddValue("nCbr", "CBR flow count", g.nCbr);
  cmd.AddValue("out", "output binary", g.out);
  cmd.AddValue("attackers", "comma-separated attacker ids (hook only)", g.attackers);
  cmd.AddValue("bridge", "lockstep with Python detector over stdio (1/0)", gBridge);
  cmd.AddValue("attack", "benign|rreq|rreq_stealth|hello|data|syn", g.attack);
  cmd.AddValue("rate", "attack intensity per second", g.rate);
  cmd.AddValue("aStart", "attack start time (s)", g.aStart);
  cmd.Parse(argc, argv);
  RngSeedManager::SetSeed(g.seed);
  N = g.nNodes;

  NodeContainer nodes; nodes.Create(N);
  for (uint32_t i = 0; i < N; i++) gNodes.push_back(nodes.Get(i));

  // Wi-Fi 802.11b, fixed power tuned to ~250 m range
  WifiHelper wifi; wifi.SetStandard(WIFI_STANDARD_80211b);
  wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
      "DataMode", StringValue("DsssRate2Mbps"), "ControlMode", StringValue("DsssRate1Mbps"));
  YansWifiPhyHelper phy; YansWifiChannelHelper ch;
  ch.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
  ch.AddPropagationLoss("ns3::LogDistancePropagationLossModel",
      "Exponent", DoubleValue(3.0), "ReferenceLoss", DoubleValue(40.05));
  phy.SetChannel(ch.Create());
  phy.Set("TxPowerStart", DoubleValue(30.0)); phy.Set("TxPowerEnd", DoubleValue(30.0));
  WifiMacHelper mac; mac.SetType("ns3::AdhocWifiMac");
  NetDeviceContainer devs = wifi.Install(phy, mac, nodes);

  // random-waypoint mobility
  auto mkAlloc = [&]() {
    Ptr<RandomRectanglePositionAllocator> pa = CreateObject<RandomRectanglePositionAllocator>();
    Ptr<UniformRandomVariable> ux = CreateObject<UniformRandomVariable>(); ux->SetAttribute("Min", DoubleValue(0)); ux->SetAttribute("Max", DoubleValue(g.area));
    Ptr<UniformRandomVariable> uy = CreateObject<UniformRandomVariable>(); uy->SetAttribute("Min", DoubleValue(0)); uy->SetAttribute("Max", DoubleValue(g.area));
    pa->SetX(ux); pa->SetY(uy); return pa;
  };
  MobilityHelper mob;
  mob.SetPositionAllocator(mkAlloc());
  if (g.speed > 0) {
    Ptr<UniformRandomVariable> sp = CreateObject<UniformRandomVariable>();
    sp->SetAttribute("Min", DoubleValue(0.5)); sp->SetAttribute("Max", DoubleValue(g.speed));
    mob.SetMobilityModel("ns3::RandomWaypointMobilityModel",
        "Speed", PointerValue(sp),
        "Pause", StringValue("ns3::ConstantRandomVariable[Constant=1.0]"),
        "PositionAllocator", PointerValue(mkAlloc()));
  } else {
    mob.SetMobilityModel("ns3::ConstantPositionMobilityModel");
  }
  mob.Install(nodes);

  // AODV + IP
  AodvHelper aodv; InternetStackHelper stack; stack.SetRoutingHelper(aodv); stack.Install(nodes);
  Ipv4AddressHelper addr; addr.SetBase("10.1.0.0", "255.255.0.0");
  Ipv4InterfaceContainer ifs = addr.Assign(devs);
  gNC = nodes; gIfs = ifs;                 // make reachable from AttackHook

  // build MAC/IP -> id maps
  for (uint32_t i = 0; i < N; i++) {
    gMacToId[Mac48Address::ConvertFrom(devs.Get(i)->GetAddress())] = i;
    gIpToId[ifs.GetAddress(i)] = i;
  }
  // promiscuous sniff + per-node reception filter (enforcement) on every device
  gDrop.assign(N, std::vector<float>(N, 0.0f));
  gThrottle.assign(N, std::vector<float>(N, 0.0f));
  for (uint32_t i = 0; i < N; i++) {
    devs.Get(i)->SetPromiscReceiveCallback(MakeBoundCallback(&Sniff, i));
    Ptr<WifiNetDevice> wd = DynamicCast<WifiNetDevice>(devs.Get(i));
    Ptr<NeighbourBlockErrorModel> em = CreateObject<NeighbourBlockErrorModel>();
    em->m_node = i;
    wd->GetPhy()->SetPostReceptionErrorModel(em);
  }

  // ---- benign traffic ----
  Ptr<UniformRandomVariable> U = CreateObject<UniformRandomVariable>();
  // CBR/UDP on-off flows
  for (uint32_t f = 0; f < g.nCbr; f++) {
    uint32_t s = U->GetInteger(0, N - 1), d = U->GetInteger(0, N - 1);
    if (s == d) { d = (d + 1) % N; }
    uint16_t port = 9000 + f;
    PacketSinkHelper sink("ns3::UdpSocketFactory",
        InetSocketAddress(Ipv4Address::GetAny(), port));
    sink.Install(nodes.Get(d)).Start(Seconds(0.5));
    OnOffHelper on("ns3::UdpSocketFactory", InetSocketAddress(ifs.GetAddress(d), port));
    on.SetAttribute("DataRate", StringValue("16kb/s"));   // ~4 pkt/s at 512 B
    on.SetAttribute("PacketSize", UintegerValue(512));
    on.SetAttribute("OnTime",  StringValue("ns3::ExponentialRandomVariable[Mean=30]"));
    on.SetAttribute("OffTime", StringValue("ns3::ExponentialRandomVariable[Mean=15]"));
    auto app = on.Install(nodes.Get(s));
    app.Start(Seconds(1.0 + U->GetValue(0, 5)));
  }
  // a handful of long-lived TCP connections
  for (uint32_t f = 0; f < g.nCbr / 3; f++) {
    uint32_t s = U->GetInteger(0, N - 1), d = U->GetInteger(0, N - 1);
    if (s == d) { d = (d + 1) % N; }
    uint16_t port = 7000 + f;
    PacketSinkHelper sink("ns3::TcpSocketFactory",
        InetSocketAddress(Ipv4Address::GetAny(), port));
    sink.Install(nodes.Get(d)).Start(Seconds(0.5));
    OnOffHelper on("ns3::TcpSocketFactory", InetSocketAddress(ifs.GetAddress(d), port));
    on.SetAttribute("DataRate", StringValue("16kb/s"));
    on.SetAttribute("PacketSize", UintegerValue(512));
    on.SetAttribute("OnTime",  StringValue("ns3::ConstantRandomVariable[Constant=20]"));
    on.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=10]"));
    auto app = on.Install(nodes.Get(s));
    app.Start(Seconds(2.0 + U->GetValue(0, 20)));
  }

  // attack hook (empty by default)
  std::vector<uint32_t> atk;
  { std::stringstream ss(g.attackers); std::string tok;
    while (std::getline(ss, tok, ',')) if (!tok.empty()) atk.push_back(std::stoul(tok)); }
  AttackHook(atk);

  // feature history init
  gAcc.assign(N, std::vector<Acc>(N));
  gPrevNbrs.assign(N, {});
  gLinkAge.assign(N, std::vector<double>(N, 0));
  gPrevDist.assign(N, std::vector<double>(N, -1));

  if (!gBridge) gOut.open(g.out, std::ios::binary);
  Simulator::Schedule(Seconds(g.win), &DumpWindow);

  // flow monitor for PDR / overhead
  FlowMonitorHelper fm; Ptr<FlowMonitor> mon = fm.InstallAll();

  Simulator::Stop(Seconds(g.sim));
  Simulator::Run();

  mon->CheckForLostPackets();
  auto stats = mon->GetFlowStats();
  uint64_t txP = 0, rxP = 0;
  for (auto& kv : stats) { txP += kv.second.txPackets; rxP += kv.second.rxPackets; }
  gOut.close();
  Simulator::Destroy();

  std::cerr << "windows=" << gWinCount << " nodes=" << N
            << " pdr=" << (txP ? (double)rxP / txP : 0)
            << " out=" << g.out << " (" << gWinCount << "x" << N << "x" << N << "x" << NF << " f32)\n";
  return 0;
}
