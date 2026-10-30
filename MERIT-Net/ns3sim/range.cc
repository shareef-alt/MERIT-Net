#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
using namespace ns3;
int main(int argc, char** argv){
  double ptx = atof(argv[1]);
  for (double d : {220.0, 240.0, 250.0, 260.0, 280.0}) {
    NodeContainer n; n.Create(2);
    WifiHelper wifi; wifi.SetStandard(WIFI_STANDARD_80211b);
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager","DataMode",StringValue("DsssRate2Mbps"),"ControlMode",StringValue("DsssRate1Mbps"));
    YansWifiPhyHelper phy; YansWifiChannelHelper ch;
    ch.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
    ch.AddPropagationLoss("ns3::LogDistancePropagationLossModel","Exponent",DoubleValue(3.0),"ReferenceLoss",DoubleValue(40.05));
    phy.SetChannel(ch.Create());
    phy.Set("TxPowerStart",DoubleValue(ptx)); phy.Set("TxPowerEnd",DoubleValue(ptx));
    WifiMacHelper mac; mac.SetType("ns3::AdhocWifiMac");
    NetDeviceContainer dv = wifi.Install(phy, mac, n);
    MobilityHelper m; auto pa = CreateObject<ListPositionAllocator>(); pa->Add(Vector(0,0,0)); pa->Add(Vector(d,0,0));
    m.SetPositionAllocator(pa); m.Install(n);
    InternetStackHelper st; st.Install(n);
    Ipv4AddressHelper a; a.SetBase("10.1.0.0","255.255.0.0"); a.Assign(dv);
    Ptr<Socket> rx = Socket::CreateSocket(n.Get(1), UdpSocketFactory::GetTypeId()); rx->Bind(InetSocketAddress(Ipv4Address::GetAny(), 9));
    uint32_t got=0; rx->SetRecvCallback(MakeBoundCallback(+[](uint32_t* g, Ptr<Socket> s){ while(s->Recv()) (*g)++; }, &got));
    Ptr<Socket> tx = Socket::CreateSocket(n.Get(0), UdpSocketFactory::GetTypeId()); tx->SetAllowBroadcast(true);
    tx->Connect(InetSocketAddress(Ipv4Address("255.255.255.255"), 9));
    for (int k=0;k<200;k++) Simulator::Schedule(Seconds(1+k*0.05), [tx](){ tx->Send(Create<Packet>(512)); });
    Simulator::Stop(Seconds(12)); Simulator::Run(); Simulator::Destroy();
    std::cout << "ptx " << ptx << " d " << d << " rx " << got << "/200" << std::endl;
  }
}
