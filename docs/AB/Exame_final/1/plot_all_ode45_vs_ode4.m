
%%
figure(1)

subplot(231)
plot(Tsol,Xsol(:,1))
hold all
plot(Tsol_ode4,Xsol_ode4(:,1))
xlabel('t [s]')
ylabel('u [ft/s]')

subplot(232)
plot(Tsol,Xsol(:,3))
hold all
plot(Tsol_ode4,Xsol_ode4(:,3))
xlabel('t [s]')
ylabel('w [ft/s]')

subplot(233)
plot(Tsol,Xsol(:,5)*180/pi)
hold all
plot(Tsol_ode4,Xsol_ode4(:,5)*180/pi)
xlabel('t [s]')
ylabel('q [deg/s]')

subplot(234)
plot(Tsol,Xsol(:,11)*180/pi)
hold all
plot(Tsol_ode4,Xsol_ode4(:,11)*180/pi)
xlabel('t [s]')
ylabel('\theta [deg]')

subplot(235)
plot(Tsol,Xsol(:,7))
hold all
plot(Tsol_ode4,Xsol_ode4(:,7))
xlabel('t [s]')
ylabel('x [ft]')

subplot(236)
plot(Tsol,Xsol(:,9))
hold all
plot(Tsol_ode4,Xsol_ode4(:,9))
xlabel('t [s]')
ylabel('z [ft]')

legend('ode45','ode4','Location','Best')

%%
figure(2)

subplot(231)
plot(Tsol,Xsol(:,2))
hold all
plot(Tsol_ode4,Xsol_ode4(:,2))
xlabel('t [s]')
ylabel('v [ft/s]')

subplot(232)
plot(Tsol,Xsol(:,10)*180/pi)
hold all
plot(Tsol_ode4,Xsol_ode4(:,10)*180/pi)
xlabel('t [s]')
ylabel('\phi [deg]')

subplot(233)
plot(Tsol,Xsol(:,4)*180/pi)
hold all
plot(Tsol_ode4,Xsol_ode4(:,4)*180/pi)
xlabel('t [s]')
ylabel('p [deg/s]')

subplot(234)
plot(Tsol,Xsol(:,6)*180/pi)
hold all
plot(Tsol_ode4,Xsol_ode4(:,6)*180/pi)
xlabel('t [s]')
ylabel('r [deg/s]')

subplot(235)
plot(Tsol,Xsol(:,12)*180/pi)
hold all
plot(Tsol_ode4,Xsol_ode4(:,12)*180/pi)
xlabel('t [s]')
ylabel('\psi [deg]')

subplot(236)
plot(Tsol,Xsol(:,8))
hold all
plot(Tsol_ode4,Xsol_ode4(:,8))
xlabel('t [s]')
ylabel('y [ft]')

legend('ode45','ode4','Location','Best')

%%
figure(3)
plot3(Xsol(:,8),Xsol(:,7),-Xsol(:,9))
hold all
plot3(Xsol_ode4(:,8),Xsol_ode4(:,7),-Xsol_ode4(:,9))
xlabel('y [ft]')
ylabel('x [ft]')
zlabel('H [ft]')
grid on
axis equal

legend('ode45','ode4','Location','Best')

%%
figure(4)

subplot(221)
plot(Tsol,Usol(:,1)*100)
hold all
plot(Tsol_ode4,Usol_ode4(:,1)*100)
xlabel('t [s]')
ylabel('\delta_t [%]')

subplot(222)
plot(Tsol,Usol(:,2))
hold all
plot(Tsol_ode4,Usol_ode4(:,2))
xlabel('t [s]')
ylabel('\delta_e [deg]')

subplot(223)
plot(Tsol,Usol(:,3))
hold all
plot(Tsol_ode4,Usol_ode4(:,3))
xlabel('t [s]')
ylabel('\delta_a [deg]')

subplot(224)
plot(Tsol,Usol(:,4))
hold all
plot(Tsol_ode4,Usol_ode4(:,4))
xlabel('t [s]')
ylabel('\delta_r [deg]')

legend('ode45','ode4','Location','Best')

%%
figure(5)

subplot(231)
plot(Tsol,Ysol(:,1))
hold all
plot(Tsol_ode4,Ysol_ode4(:,1))
xlabel('t [s]')
ylabel('V [ft/s]')

subplot(232)
plot(Tsol,Ysol(:,2))
hold all
plot(Tsol_ode4,Ysol_ode4(:,2))
xlabel('t [s]')
ylabel('\alpha [deg]')

subplot(233)
plot(Tsol,Ysol(:,7))
hold all
plot(Tsol_ode4,Ysol_ode4(:,7))
xlabel('t [s]')
ylabel('\beta [deg]')

subplot(234)
plot(Tsol,Xsol(:,13))
hold all
plot(Tsol_ode4,Xsol_ode4(:,13))
xlabel('t [s]')
ylabel('Power [%]')

subplot(235)
plot(Tsol,Ysol(:,14))
hold all
plot(Tsol_ode4,Ysol_ode4(:,14))
xlabel('t [s]')
ylabel('n_{y,pilot}')

subplot(236)
plot(Tsol,Ysol(:,15))
hold all
plot(Tsol_ode4,Ysol_ode4(:,15))
xlabel('t [s]')
ylabel('n_{z,pilot}')

legend('ode45','ode4','Location','Best')