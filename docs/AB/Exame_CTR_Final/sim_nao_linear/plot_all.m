figure(1)

subplot(231)
plot(Tsol,Xsol(:,1))
xlabel('t [s]')
ylabel('u [ft/s]')

subplot(232)
plot(Tsol,Xsol(:,3))
xlabel('t [s]')
ylabel('w [ft/s]')

subplot(233)
plot(Tsol,Xsol(:,5)*180/pi)
xlabel('t [s]')
ylabel('q [deg/s]')

subplot(234)
plot(Tsol,Xsol(:,11)*180/pi)
xlabel('t [s]')
ylabel('\theta [deg]')

subplot(235)
plot(Tsol,Xsol(:,7))
xlabel('t [s]')
ylabel('x [ft]')

subplot(236)
plot(Tsol,Xsol(:,9))
xlabel('t [s]')
ylabel('z [ft]')

%%
fig15 = figure(1)
% export to pdf
set(fig15,'Units','Inches');
pos = get(fig15,'Position');
set(fig15,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(fig15,'fig15','-dpdf','-r0')
%%
figure(2)

subplot(231)
plot(Tsol,Xsol(:,2))
xlabel('t [s]')
ylabel('v [ft/s]')

subplot(232)
plot(Tsol,Xsol(:,10)*180/pi)
xlabel('t [s]')
ylabel('\phi [deg]')

subplot(233)
plot(Tsol,Xsol(:,4)*180/pi)
xlabel('t [s]')
ylabel('p [deg/s]')

subplot(234)
plot(Tsol,Xsol(:,6)*180/pi)
xlabel('t [s]')
ylabel('r [deg/s]')

subplot(235)
plot(Tsol,Xsol(:,12)*180/pi)
xlabel('t [s]')
ylabel('\psi [deg]')

subplot(236)
plot(Tsol,Xsol(:,8))
xlabel('t [s]')
ylabel('y [ft]')
%%
fig16 = figure(2)
% export to pdf
set(fig16,'Units','Inches');
pos = get(fig16,'Position');
set(fig16,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(fig16,'fig16','-dpdf','-r0')
%%
figure(3)
plot3(Xsol(:,8),Xsol(:,7),-Xsol(:,9),'r-')
xlabel('y [ft]')
ylabel('x [ft]')
zlabel('H [ft]')
grid on
%%
fig17 = figure(3)
% export to pdf
set(fig17,'Units','Inches');
pos = get(fig17,'Position');
set(fig17,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(fig17,'fig22','-dpdf','-r0')

%%
figure(4)

subplot(221)
plot(Tsol,Usol(:,1)*100)
xlabel('t [s]')
ylabel('\delta_t [%]')

subplot(222)
plot(Tsol,Usol(:,2))
xlabel('t [s]')
ylabel('\delta_e [deg]')

subplot(223)
plot(Tsol,Usol(:,3))
xlabel('t [s]')
ylabel('\delta_a [deg]')

subplot(224)
plot(Tsol,Usol(:,4))
xlabel('t [s]')
ylabel('\delta_r [deg]')
%%
fig18 = figure(4)
% export to pdf
set(fig18,'Units','Inches');
pos = get(fig18,'Position');
set(fig18,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(fig18,'fig18','-dpdf','-r0')
%%

figure(5)

subplot(231)
plot(Tsol,Ysol(:,1))
hold all
% plot(Tsol_ode4,Ysol_ode4(:,1))
xlabel('t [s]')
ylabel('V [ft/s]')

subplot(232)
plot(Tsol,Ysol(:,2))
hold all
% plot(Tsol_ode4,Ysol_ode4(:,2))
xlabel('t [s]')
ylabel('\alpha [deg]')

subplot(233)
plot(Tsol,Ysol(:,7))
hold all
% plot(Tsol_ode4,Ysol_ode4(:,7))
xlabel('t [s]')
ylabel('\beta [deg]')

subplot(234)
plot(Tsol,Xsol(:,13))
hold all
% plot(Tsol_ode4,Xsol_ode4(:,13))
xlabel('t [s]')
ylabel('Power [%]')

subplot(235)
plot(Tsol,Ysol(:,14))
hold all
% plot(Tsol_ode4,Ysol_ode4(:,14))
xlabel('t [s]')
ylabel('n_{y,pilot}')

subplot(236)
plot(Tsol,Ysol(:,15))
hold all
% plot(Tsol_ode4,Ysol_ode4(:,15))
xlabel('t [s]')
ylabel('n_{z,pilot}')
%%
fig19 = figure(5)
% export to pdf
set(fig19,'Units','Inches');
pos = get(fig19,'Position');
set(fig19,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(fig19,'fig19','-dpdf','-r0')
%%

figure(6)
plot(Tsol,Ysol(:,28))
hold all
xlabel('t [s]')
ylabel('n_{z,pilot}')
%%
fig20 = figure(6)
% export to pdf
set(fig20,'Units','Inches');
pos = get(fig20,'Position');
set(fig20,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(fig20,'fig20','-dpdf','-r0')


