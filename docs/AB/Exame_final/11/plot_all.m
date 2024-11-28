
%%
figure(1)

subplot(231)
plot(Tsol,Ysol(:,36))
xlabel('t [s]')
ylabel('u [m/s]')

subplot(232)
plot(Tsol,Ysol(:,38))
xlabel('t [s]')
ylabel('w [m/s]')

subplot(233)
plot(Tsol,Ysol(:,3))
xlabel('t [s]')
ylabel('q [deg/s]')

subplot(234)
plot(Tsol,Ysol(:,4))
xlabel('t [s]')
ylabel('\theta [deg]')

subplot(235)
plot(Tsol,Ysol(:,6))
xlabel('t [s]')
ylabel('x [m]')

subplot(236)
plot(Tsol,Ysol(:,5))
xlabel('t [s]')
ylabel('z [m]')
% 
% fig1 = figure(1)
% % export to pdf
% set(fig1,'Units','Inches');
% pos = get(fig1,'Position');
% set(fig1,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
% print(fig1,'1_1','-dpdf','-r0')
%%
figure(2)

subplot(231)
plot(Tsol,Ysol(:,37))
xlabel('t [s]')
ylabel('v [m/s]')

subplot(232)
plot(Tsol,Xsol(:,8))
xlabel('t [s]')
ylabel('\phi [deg]')

subplot(233)
plot(Tsol,Ysol(:,9))
xlabel('t [s]')
ylabel('p [deg/s]')

subplot(234)
plot(Tsol,Ysol(:,10))
xlabel('t [s]')
ylabel('r [deg/s]')

subplot(235)
plot(Tsol,Ysol(:,11))
xlabel('t [s]')
ylabel('\psi [deg]')

subplot(236)
plot(Tsol,Ysol(:,12))
xlabel('t [s]')
ylabel('y [m]')

% fig2 = figure(2)
% % export to pdf
% set(fig2,'Units','Inches');
% pos = get(fig2,'Position');
% set(fig2,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
% print(fig2,'1_2','-dpdf','-r0')
%%
figure(3)
plot3(Xsol(:,12),Xsol(:,6),Xsol(:,5))
xlabel('y [m]')
ylabel('x [m]')
zlabel('H [m]')
grid on
axis equal

% fig3 = figure(3)
% % export to pdf
% set(fig3,'Units','Inches');
% pos = get(fig3,'Position');
% set(fig3,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
% print(fig3,'1_3','-dpdf','-r0')

%%
figure(4)

subplot(231)
plot(Tsol,Usol(:,1))
xlabel('t [s]')
ylabel('\delta_tle [N]')

subplot(232)
plot(Tsol,Usol(:,2))
xlabel('t [s]')
ylabel('\delta_tre [N]')

subplot(233)
plot(Tsol,Usol(:,3))
xlabel('t [s]')
ylabel('\delta_ih [deg]')

subplot(234)
plot(Tsol,Usol(:,4))
xlabel('t [s]')
ylabel('\delta_e [deg]')

subplot(235)
plot(Tsol,Usol(:,5))
xlabel('t [s]')
ylabel('\delta_a [deg]')

subplot(236)
plot(Tsol,Usol(:,6))
xlabel('t [s]')
ylabel('\delta_r [deg]')
% 
% fig4 = figure(4)
% % export to pdf
% set(fig4,'Units','Inches');
% pos = get(fig4,'Position');
% set(fig4,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
% print(fig4,'1_4','-dpdf','-r0')
%%
figure(5)

subplot(231)
plot(Tsol,Ysol(:,1))
xlabel('t [s]')
ylabel('V [m/s]')

subplot(232)
plot(Tsol,Ysol(:,2))
xlabel('t [s]')
ylabel('\alpha [deg]')

subplot(233)
plot(Tsol,Ysol(:,7))
xlabel('t [s]')
ylabel('\beta [deg]')

subplot(234)
plot(Tsol,Usol(:,1)+Usol(:,2))
xlabel('t [s]')
ylabel('T [N]')

subplot(235)
plot(Tsol,Ysol(:,14))
xlabel('t [s]')
ylabel('n_{y,pilot}')

subplot(236)
plot(Tsol,Ysol(:,15))
xlabel('t [s]')
ylabel('n_{z,pilot}')

% fig5 = figure(5)
% % export to pdf
% set(fig5,'Units','Inches');
% pos = get(fig5,'Position');
% set(fig5,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
% print(fig5,'1_5','-dpdf','-r0')
%% 

figure(6)

subplot(231)
plot(Tsol,Ysol(:,2))
xlabel('t [s]')
ylabel('\alpha [deg]')

subplot(232)
plot(Tsol,Usol(:,3))
xlabel('t [s]')
ylabel('i_h [deg]')

subplot(233)
plot(Tsol,Ysol(:,28))
xlabel('t [s]')
ylabel('D [N]')

subplot(234)
plot(Tsol,Usol(:,1)+Usol(:,2))
xlabel('t [s]')
ylabel('T_{total} [N]')

subplot(235)
plot(Tsol,Usol(:,1))
xlabel('t [s]')
ylabel('T_{le} [N]')

subplot(236)
plot(Tsol,Usol(:,2))
xlabel('t [s]')
ylabel('T_{re} [N]')


% fig6 = figure(6)
% % export to pdf
% set(fig6,'Units','Inches');
% pos = get(fig6,'Position');
% set(fig6,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
% print(fig6,'1_6','-dpdf','-r0')
