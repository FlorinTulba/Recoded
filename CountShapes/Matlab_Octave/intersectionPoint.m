%{
    Part of the CountShapes project.

    @2017 Florin Tulba (florintulba@yahoo.com)
%}

function [ x y ] = intersectionPoint(Ra, thetaA, Rb, thetaB)
%{
INTERSECTIONPOINT Return the coordinates of the intersection of 2 lines

The 2 lines are specified by their Hesse normal form:
   Ra = x*cos(thetaA) + y*sin(thetaA)
   Rb = x*cos(thetaB) + y*sin(thetaB)

After doing the math, the intersection's coordinates (x,y) are:
    x = (Ra*sin(thetaB) - Rb*sin(thetaA)) / sin(thetaB - thetaA)
    y = (Rb*cos(thetaA) - Ra*cos(thetaB)) / sin(thetaB - thetaA)

Obviously, when the lines are parallel (thetaA == thetaB),
there is no solution or the lines coincide.
%}
    if abs(thetaA - thetaB) < 1e-8
        x = NaN; y = NaN;
        return
    end
    
    % transform degrees in radians
    thetaA = pi * thetaA / 180;
    thetaB = pi * thetaB / 180;
    
    divider = sin(thetaB - thetaA);
    x = (Ra*sin(thetaB) - Rb*sin(thetaA)) / divider;
    y = (Rb*cos(thetaA) - Ra*cos(thetaB)) / divider;
end

