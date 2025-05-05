sim=require'sim'

function sysCall_init()
    -- Los joints de los dedos están directamente bajo NiryoOneGripper
    leftFinger = sim.getObject('../leftJoint1')
    rightFinger = sim.getObject('../rightJoint1')

    -- Límites de apertura (ajusta si tu pinza abre más o menos)
    minPos = 0         -- completamente cerrado
    maxPos = 0.02      -- completamente abierto
end

function sysCall_actuation()
    -- Control continuo desde Python
    local opening = sim.getFloatSignal('gripper_opening')
    if opening ~= nil then
        local pos = minPos + (maxPos - minPos) * opening
        sim.setJointTargetPosition(leftFinger, pos)
        sim.setJointTargetPosition(rightFinger, pos)
    end
end