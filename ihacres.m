function [Q_pred, uk_pred, xkq_pred, xks_pred]=ihacres(rain,temp,par)

%%%%%This is a IHACRES code develop by Dr. Rajarshi Das Bhowmik%%%%%
%%%%%Indian Institute of Science%%%%%%%%%%%%
%%%%Date Oct 17, 2023%%%%%%%%

rk=rain;%%variable
tk=temp;%%variable
[tt,nn]=size(rain);%%size of the data

%unpacking the parameters
c=par(1);
tau=par(2);
f=par(3);
tauq=par(4);                                                                    
taus=par(5);
vq=par(6);

%%%random initializations
xkq(1,1)=0.1;
xks(1,1)=0.1;
sk(1,1)=0.01;

%%%%%%%%%Runoff generation%%%%%%%
for i=1:tt%%%calculating wetness index for all observations
sk(i+1,1)=(rk(i,1)/c)+(1-(1/(tau*exp(20-tk(i,1))*f)))*sk(i,1);
if sk(i+1,1)<0 %%%negetive wetness index is not possible
    sk(i+1,1)=0.01;%%%just a random small value
end

if rk(i,1)>0.01 %%%rainfall threshold
    uk=sk(i+1,1)*rk(i,1);%%%runoff
else
    uk=0;
end
if uk>rk(i,1)%%%runoff cant exceed rainfall
    uk=rk(i,1);
end
ukk(i,1) = uk;

%%%Runoff components%%%%
xkq(i+1,1)=exp(-1/tauq)*xkq(i,1)+vq*(1-exp(-1/tauq))*uk;%%quick flow
xks(i+1,1)=exp(-1/taus)*xks(i,1)+(1-vq)*(1-exp(-1/taus))*uk;%slow flow
if xkq(i+1,1)<0%%%to ensure no negetive value
    xkq(i+1,1)=0;
end
if xks(i+1,1)<0 %%%to ensure no negetive value
    xks(i+1,1)=0;
end
qq(i,1)=xks(i+1,1)+xkq(i+1,1);
end
uk_pred = ukk;
xkq_pred = xkq;
xks_pred = xks;
Q_pred=qq;%%discharge

end